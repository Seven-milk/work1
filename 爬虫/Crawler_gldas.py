# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# crawler for getting GLDAS nc4 file from urls.txt, fail due to the access
import requests
import os
import re
import aiohttp
import asyncio

home = "H:/data_zxd/GLDAS"
url_path = [txt for txt in os.listdir(home) if txt[-4:] == ".txt"]
save_path = "H:/data_zxd/GLDAS/GLDAS_NOAH"

# read url.txt
with open(os.path.join(home, url_path[0])) as f:
    urls = f.read()
urls = urls.split("\n")

# save name, e.g. GLDAS_NOAH025_3H.A19480101.0300.020.nc4
save_name = [re.search(r'GLDAS_NOAH025_3H.A\d{8}\.\d{4}\.020\.nc4', url)[0] for url in urls[1:]]

# url[0]: get pdf
r = requests.get(urls[0])
with open(f'{save_path}/GLDAS2.pdf', 'wb') as f:
    f.write(r.content)
urls = urls[1:]  # delete url of pdf


# without async
urls = urls[:5]


def fetch_no_async():
    ''' fetch with out async '''
    for i in range(len(urls)):
        r = requests.get(urls[i])
        with open(f'{save_path}/{save_name[i]}', 'wb') as f:
            f.write(r.content)


# async
async def fetch_async(sem, url):
    print(f"start{url}")
    async with sem:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                save_name_ = re.search(r'GLDAS_NOAH025_3H.A\d{8}\.\d{4}\.020\.nc4', url)[0]
                with open(f'{save_path}/{save_name_}', 'wb') as f:
                    chunk = await response.content.read()
                    f.write(chunk)


async def main():
    sem = asyncio.Semaphore(3)  # limit number of coroutines
    tasks = [asyncio.create_task(fetch_async(sem, url)) for url in urls]
    await asyncio.wait(tasks)


if __name__ == "__main__":
    asyncio.run(main())
    # fetch_no_async()
