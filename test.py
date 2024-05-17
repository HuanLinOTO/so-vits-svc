import asyncio
import subprocess

import logger


async def exec_it(command, callback):
    accumulated_output = ""
    try:
        # command = 'python -c "import time; [print(i) or time.sleep(1) for i in range(1, 6)]"'
        result = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )
        for line in result.stdout:
            callback(line)
        result.communicate()
    except subprocess.CalledProcessError as e:
        result = e.output
        accumulated_output += f"Error: {result}\n"
        callback(accumulated_output)


def callback_demo(output):
    if "processed 1 file" in output:
        print("callback_demo: processed 1 file")


async def main():
    filelists = ["filelists/train.txt", "filelists/val.txt"]
    tasks = []
    for filelist in filelists:
        tasks.append(
            exec_it(
                f"python preprocess_chunk.py --filelist {filelist}",
                callback=callback_demo,
            )
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
