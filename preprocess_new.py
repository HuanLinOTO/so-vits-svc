import argparse
import asyncio
import os
import subprocess
import time

import torch

import logger


async def exec_it(command, callback):
    accumulated_output = ""
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            callback(str(line))

        await process.wait()
    except subprocess.CalledProcessError as e:
        result = e.output
        accumulated_output += f"Error: {result}\n"
        callback(accumulated_output)


def _callback(name):
    def real_cb(output):
        print(output)
        if "processed 1 file" in output:
            print(f"{name}: processed 1 file")

    return real_cb


async def main(device, f0p, num_processes, use_diff):
    # print(speech_encoder)
    logger.info("Using device: " + str(device))

    logger.info("Using f0 extractor: " + f0p)

    # 读取 train 和 val 的 filelist 分成 num_processes 块，输出到filelists/{timestramp}/chunk-n.txt 文件夹不存在时创建
    # Create the directory for filelists
    timestamp = str(int(time.time()))
    tmp_path = f"filelists/{timestamp}"
    os.makedirs(tmp_path, exist_ok=True)

    # Read train and val filelists
    with open(args.train_filelist, "r", encoding="utf-8") as train_file:
        train_lines = train_file.readlines()

    with open(args.val_filelist, "r", encoding="utf-8") as val_file:
        val_lines = val_file.readlines()

    # Calculate the number of lines per chunk
    num_train_lines = len(train_lines)
    num_val_lines = len(val_lines)
    lines_per_chunk = (num_train_lines + num_val_lines) // args.num_processes

    # Split train filelist into chunks
    for i in range(args.num_processes):
        start_idx = i * lines_per_chunk
        end_idx = start_idx + lines_per_chunk

        train_chunk = train_lines[start_idx:end_idx]

        with open(
            f"filelists/{timestamp}/chunk-{i}.txt", "w", encoding="utf-8"
        ) as chunk_file:
            chunk_file.writelines(train_chunk)

    # Split val filelist into chunks
    for i in range(args.num_processes):
        start_idx = i * lines_per_chunk
        end_idx = start_idx + lines_per_chunk

        val_chunk = val_lines[start_idx:end_idx]

        with open(
            f"filelists/{timestamp}/chunk-{i}.txt", "a", encoding="utf-8"
        ) as chunk_file:
            chunk_file.writelines(val_chunk)

    filelists = []
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            if file.endswith(".txt"):
                filelists.append(os.path.join(root, file))
    tasks = []
    print(filelists)
    for filelist in filelists:
        command = f"python preprocess_chunk.py --filelist {filelist}"
        if use_diff:
            command += " --use_diff"
        tasks.append(
            exec_it(
                command,
                callback=_callback(filelist),
            )
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument(
        "--train-filelist",
        type=str,
        default="filelists/train.txt",
        help="path to val filelist.txt",
    )
    parser.add_argument(
        "--val-filelist",
        type=str,
        default="filelists/val.txt",
        help="path to val filelist.txt",
    )
    parser.add_argument(
        "--use_diff", action="store_true", help="Whether to use the diffusion model"
    )
    parser.add_argument(
        "--f0_predictor",
        type=str,
        default="rmvpe",
        help="Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="You are advised to set the number of processes to the same as the number of CPU cores",
    )

    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_processes = args.num_processes
    use_diff = args.use_diff

    asyncio.run(main(device, f0p, num_processes, use_diff))
