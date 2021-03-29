# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# crawler for getting GLDAS nc4 file from urls.txt, fail due to the access(fixed by cookie)
import requests
import os
import re
import aiohttp
import asyncio
from pathos.multiprocessing import ProcessingPool as Pool


home = "H:/data_zxd/GLDAS/GLDAS_NOAH"
url_path = [txt for txt in os.listdir(home) if txt[-4:] == ".txt"]
save_path = "H:/data_zxd/GLDAS/GLDAS_NOAH"

# read url.txt
with open(os.path.join(home, url_path[0])) as f:
    urls = f.read()
urls = urls.split("\n")

# save name, e.g. GLDAS_NOAH025_3H.A19480101.0300.020.nc4
save_name = [re.search(r'GLDAS_NOAH025_3H.A\d{8}\.\d{4}\.020\.nc4', url)[0] for url in urls[1:]]

# url[0]: get pdf
# r = requests.get(urls[0])
# with open(f'{save_path}/GLDAS2.pdf', 'wb') as f:
#     f.write(r.content)
urls = urls[1:]  # delete url of pdf

# cookie
cookie = {"_ce.s": "v~aiJDYxoSAkD_8TJPGNtQbQTxj78~ir~1~nvisits_null~1~validSession_null~1",
     "_ga": "GA1.4.1469818823.1578554237",
     "_gat_GSA_ENOR0": "1",
     "_gid": "GA1.4.769268356.1605496087",
     "119161751904181415183710131262": "s:y-MflyZIqKV_pWWkJ268EIVZ527R-AhK.tJcnKmUJLhMENvfFB60srqHRt03UvcsJPF4RbmgLrAU",
     "413371814191509516817101261112": "s:qMLvn15_1M3ogo1i0x9mvOMIaraAqbGR.p+jysFSfDYBqxgbcrkLu+ttCn+KjWKVAEUOnLG8nbJc",
     "nasa_gesdisc_data_archive": "cJgmphp/SKycowIuYsiFDmMCHGMgpRTH/yay0t8uhVJiy9tcCcisWA5a6JOktuixBFdA6xn9Jm+BSQ+VvPH7z4f6v2MIhO4WF5bpc/SRRv8N5tAM9XmtC8W/B1D/XoLCttXHpTuYnD2tLhbAB9ERBeTEn72lyFjRb7SwYorjfaE=",
     "urs_guid_ops": "c8b3a370-29fb-4187-a930-a35abda0592b"
     }
# header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0",
#           "Cookie": cookie}

# test
# urls = urls[:5]

# without async
def fetch_no_async():
    ''' fetch with out async '''
    for i in range(len(urls)):
        r = requests.get(urls[i], cookies=cookie)
        print(i, ":", r)
        with open(f'{save_path}/{save_name[i]}', 'wb') as f:
            f.write(r.content)


# mp
def fetch(i):
    r = requests.get(urls[i], cookies=cookie)
    print(i, ":", r)
    with open(f'{save_path}/{save_name[i]}', 'wb') as f:
        f.write(r.content)


def mp(num_pool=8):
    po = Pool(num_pool)
    for i in range(len(urls)):
        po.amap(fetch, (i,))
    po.close()
    po.join()


# async
async def fetch_async(sem, url):
    print(f"start{url}")
    async with sem:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(session, ":", response)
                save_name_ = re.search(r'GLDAS_NOAH025_3H.A\d{8}\.\d{4}\.020\.nc4', url)[0]
                with open(f'{save_path}/{save_name_}', 'wb') as f:
                    chunk = await response.content.read()
                    f.write(chunk)


async def main():
    sem = asyncio.Semaphore(3)  # limit number of coroutines
    tasks = [asyncio.create_task(fetch_async(sem, url)) for url in urls]
    await asyncio.wait(tasks)


if __name__ == "__main__":
    pass
    # asyncio.run(main())
    # fetch_no_async()
    mp()