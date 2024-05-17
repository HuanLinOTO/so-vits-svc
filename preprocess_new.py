import argparse
import os
import subprocess
import time

import torch

import logger


def exec_command(cmd, callback):
    # 执行 cmd，当进程的 stdout/stderr 输出了 processed 1 file 的时候调用 callback
    logger.info(f"Executing command: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            logger.info(output.strip())
            if "processed 1 file" in output.decode("utf-8"):
                callback()


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

        with open(f"filelists/{timestamp}/chunk-{i}.txt", "w") as chunk_file:
            chunk_file.writelines(train_chunk)

    # Split val filelist into chunks
    for i in range(args.num_processes):
        start_idx = i * lines_per_chunk
        end_idx = start_idx + lines_per_chunk

        val_chunk = val_lines[start_idx:end_idx]

        with open(f"filelists/{timestamp}/chunk-{i}.txt", "a") as chunk_file:
            chunk_file.writelines(val_chunk)
