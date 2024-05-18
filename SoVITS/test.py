import asyncio
import os
import subprocess


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


def callback_demo(name):
    def real_cb(output):
        # logger.info(output.strip())
        if "processed 1 file" in output:
            print(f"{name}: processed 1 file")

    return real_cb


async def main():
    # 扫描 filelists\1716012223 下面的 txt 变成 filelists 数组
    filelists = []
    for root, dirs, files in os.walk("filelists/1716012441"):
        for file in files:
            if file.endswith(".txt"):
                filelists.append(os.path.join(root, file))
    tasks = []
    print(filelists)
    for filelist in filelists:
        tasks.append(
            exec_it(
                f"python preprocess_chunk.py --filelist {filelist}",
                callback=callback_demo(filelist),
            )
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
