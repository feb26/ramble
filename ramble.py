#!/usr/bin/env python3
"""ramble: ローカル音声文字起こしツール

マイクから録音した音声をmlx_whisperで文字起こしし、
テキストファイルに保存 + クリップボードにコピーする。

設計判断:
- 文字起こし結果はフィラー（えーと、あのー等）を含む生テキストで出力する。
  整理・構造化は別途生成AIに任せる想定（2段パイプライン）。
- mlx_whisperは内部で30秒チャンクに分割するため、15分程度の音声も問題なく処理可能。
- モデルは初回transcribe時にHugging Faceからダウンロードされ、
  ~/.cache/huggingface/hub/ にキャッシュされる（2回目以降はネットワーク不要）。
- 待機中はinput()でブロックするためCPU消費はほぼゼロ。
  録音データ(audio)とテキスト(text)は毎ループ解放するためメモリも膨らまない。
  ただしmlx_whisperのモデル（数GB）はプロセス終了まで常駐する。
"""

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

# Whisperの入力仕様に合わせて16kHz mono float32で録音する
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_DIR = Path(__file__).parent / "output"

MODELS = [
    ("large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
    ("medium", "mlx-community/whisper-medium"),
    ("tiny", "mlx-community/whisper-tiny"),
]


def get_input_devices():
    """入力可能なオーディオデバイスの一覧を返す。[(id, name), ...]"""
    devices = sd.query_devices()
    return [
        (i, d["name"])
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def get_default_device_id():
    """システムのデフォルト入力デバイスIDを返す。"""
    return sd.default.device[0]


def select_device(current_device_id):
    """マイクデバイスを対話的に選択。選択されたデバイスIDを返す。"""
    devices = get_input_devices()
    print()
    print("  マイクデバイス:")
    for device_id, name in devices:
        marker = " (現在)" if device_id == current_device_id else ""
        print(f"    [{device_id}] {name}{marker}")
    print()

    try:
        choice = input("  番号を入力: ").strip()
        if not choice:
            return current_device_id
        chosen_id = int(choice)
        valid_ids = [d[0] for d in devices]
        if chosen_id not in valid_ids:
            print("  無効な番号です。変更しません。")
            return current_device_id
        chosen_name = next(name for did, name in devices if did == chosen_id)
        print(f"  → {chosen_name} に変更しました")
        return chosen_id
    except (ValueError, KeyboardInterrupt, EOFError):
        print()
        return current_device_id


def select_model(current_model_index):
    """モデルを対話的に選択。選択されたモデルのインデックスを返す。"""
    print()
    print("  モデル:")
    for i, (name, _) in enumerate(MODELS):
        marker = " (現在)" if i == current_model_index else ""
        print(f"    [{i}] {name}{marker}")
    print()

    try:
        choice = input("  番号を入力: ").strip()
        if not choice:
            return current_model_index
        chosen = int(choice)
        if chosen < 0 or chosen >= len(MODELS):
            print("  無効な番号です。変更しません。")
            return current_model_index
        print(f"  → {MODELS[chosen][0]} に変更しました")
        return chosen
    except (ValueError, KeyboardInterrupt, EOFError):
        print()
        return current_model_index


def get_device_name(device_id):
    """デバイスIDから名前を取得。"""
    try:
        return sd.query_devices(device_id)["name"]
    except Exception:
        return f"Device {device_id}"


def show_header(device_id, model_index):
    """現在の設定とメニューを表示。"""
    device_name = get_device_name(device_id)
    model_name = MODELS[model_index][0]
    print()
    print("=" * 50)
    print("  ramble")
    print(f"  マイク: {device_name}")
    print(f"  モデル: {model_name}")
    print("=" * 50)
    print()
    print("  [Enter] 録音開始")
    print("  [d]     マイクデバイス変更")
    print("  [m]     モデル変更")
    print("  [q]     終了")
    print()


def record(device_id):
    """録音してnumpy配列を返す。EnterまたはCtrl+Cで停止。"""
    audio_chunks = []
    is_recording = threading.Event()
    is_recording.set()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n  [audio warning: {status}]", file=sys.stderr)
        if is_recording.is_set():
            audio_chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device_id,
        callback=callback,
    )

    start_time = time.time()
    stream.start()

    def show_elapsed():
        while is_recording.is_set():
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            print(f"\r  録音中... {minutes:02d}:{seconds:02d} (Enter or Ctrl+C で停止)", end="", flush=True)
            time.sleep(0.5)

    timer_thread = threading.Thread(target=show_elapsed, daemon=True)
    timer_thread.start()

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    is_recording.clear()
    stream.stop()
    stream.close()
    timer_thread.join(timeout=1)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\r  録音完了: {minutes:02d}:{seconds:02d}                                  ")

    if not audio_chunks:
        return None

    return np.concatenate(audio_chunks, axis=0).flatten()


def transcribe(audio, model_repo):
    """mlx_whisperで文字起こし。"""
    import mlx_whisper

    print("  文字起こし中...")
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model_repo,
        language="ja",
    )
    return result["text"]


def save_and_copy(text):
    """テキストをファイルに保存し、クリップボードにコピー。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = OUTPUT_DIR / f"{timestamp}.txt"
    filepath.write_text(text, encoding="utf-8")

    try:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        print("  クリップボードにコピーしました")
    except Exception as e:
        print(f"  クリップボードへのコピーに失敗: {e}", file=sys.stderr)

    print(f"  保存先: {filepath.resolve()}")
    return filepath


def main():
    device_id = get_default_device_id()
    model_index = 0  # large-v3-turbo

    while True:
        show_header(device_id, model_index)

        try:
            choice = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  終了します")
            break

        if choice == "q":
            print("  終了します")
            break
        elif choice == "d":
            device_id = select_device(device_id)
            continue
        elif choice == "m":
            model_index = select_model(model_index)
            continue
        elif choice != "":
            print(f"  不明なコマンド: {choice}")
            continue

        # Enter: 録音開始
        audio = record(device_id)

        if audio is None or len(audio) < SAMPLE_RATE:
            print("  録音が短すぎます。もう一度試してください。")
            continue

        model_repo = MODELS[model_index][1]
        try:
            text = transcribe(audio, model_repo)
        except KeyboardInterrupt:
            print("\n  文字起こしをキャンセルしました")
            continue
        finally:
            del audio

        if not text.strip():
            print("  文字起こし結果が空でした。")
            continue

        print()
        print("  --- 文字起こし結果 ---")
        print(text)
        print("  ----------------------")
        print()

        save_and_copy(text)
        del text


if __name__ == "__main__":
    main()
