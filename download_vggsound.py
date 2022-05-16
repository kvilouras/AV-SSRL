import os
import subprocess
import csv
import time
import argparse
import multiprocessing as mp
from itertools import islice
import signal
import json


parser = argparse.ArgumentParser(description='Download VGGSound dataset')

parser.add_argument('data_path',
                    action='store',
                    nargs='?',
                    default=os.getcwd(),
                    help='Path to directory where data will be stored')

parser.add_argument('cookies_path',
                    action='store',
                    nargs='?',
                    default=None,
                    help='Path to cookies.txt file (to avoid too many requests error)')

parser.add_argument('-f',
                    '--ffmpeg',
                    dest='ffmpeg_path',
                    action='store',
                    type=str,
                    default='/usr/bin/ffmpeg',
                    help='Path to ffmpeg executable')

parser.add_argument('-fp',
                    '--ffprobe',
                    dest='ffprobe_path',
                    action='store',
                    type=str,
                    default='/usr/bin/ffprobe',
                    help='Path to ffprobe executable')

parser.add_argument('-n',
                    '--n_workers',
                    dest='n_workers',
                    action='store',
                    type=int,
                    default=mp.cpu_count(),
                    help='Number of workers used to download videos in parallel')

args = parser.parse_args()


def initializer():
    """
    Ignore SIGINT in child workers (to handle keyboard interrupt while multiprocessing)
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def callback(x):
    """
    Callback to terminate pool under certain conditions
    """
    if x:
        pool.terminate()


def download_snippet(video_id, start_time, dest_dir, ffmpeg_path, cookies_path):
    """
    Download excerpt from given Youtube video
    :param video_id: Identifier of the input Youtube video
    :param start_time: Beginning of the segment to be extracted
    :param dest_dir: Directory where the extracted segment will be stored
    :param ffmpeg_path: Path to ffmpeg
    :param cookies_path: Path to cookies.txt file (to avoid HTTP 429 error)
    :return: flag (None if the excerpt is downloaded or True if some errors occured)
    """
    global counter, retry_flag

    flag = None  # to terminate the pool if necessary

    if cookies_path is None:
        proc = subprocess.Popen(["bash", "download_video.sh", video_id, start_time, dest_dir, ffmpeg_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else:
        proc = subprocess.Popen(["bash", "download_video.sh", video_id, start_time, dest_dir, ffmpeg_path, cookies_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    try:
        stdout, stderr = proc.communicate()
    except KeyboardInterrupt:
        exit()

    if proc.returncode == 0:
        # update counter
        with counter.get_lock():
            counter.value += 1
        time.sleep(2)  # to avoid HTTP Error 429: Too Many Requests
    else:
        # error handling
        # 1) retry download if HTTP Error 503 occurs after a few seconds
        if ("http error 503" in stderr.lower() or "service is unavailable" in stderr.lower() or
                "there was a problem with the network" in stderr.lower()):
            print('Service is unavailable!')    
            if not retry_flag.value:
                time.sleep(60)
                retry_flag.value = 1  # if this occurs a 2nd time, then abort execution
                if download_snippet(video_id, start_time, dest_dir, ffmpeg_path, cookies_path) is None:
                    retry_flag.value = 0  # download completed
            else:
                flag = True
        # 2) too many requests means that Youtube forces us to stop downloading
        elif "too many requests" in stderr.lower() or "http error 429" in stderr.lower():
            print('Reached max requests!')
            flag = True
        # 3) the video is unavailable and should be removed
        else:
            print(stderr)
            # update counter
            with counter.get_lock():
                counter.value += 1

    return flag


# create directories (if necessary)
data_dir = os.path.join(args.data_path, 'vggsound_data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Original csv file can be found here: https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv
# It is expected to be located in the data directory stated in the input
assert os.path.exists(os.path.join(args.data_path, 'vggsound.csv'))
csv_path = os.path.join(args.data_path, 'vggsound.csv')

# Handle different extensions (provided that we downloaded files at least once before):
#   1) .mp4.part (do not delete + re-download),
#   2) .mp4.ytdl (skip)
#   3) .webm + .m4a + .webm.part + .temp.mp4 (delete + re-download) -- CHECK FOR DUPLICATES
#   4) .webm.ytdl (delete)
#   5) delete + re-download .mp4 files whose video stream duration is longer than expected (i.e. 10 sec) --> shows that
#      something went wrong while slicing the segment
vids = []
for dirname, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith('.mp4.ytdl'):
            continue
        elif filename.endswith('.webm.ytdl'):
            os.remove(os.path.join(dirname, filename))
        elif filename.endswith('.part'):
            p1 = filename.split('.')[0]
            try:
                vid, start = p1.split('_')
            except ValueError:
                start = p1.split('_')[-1]
                vid = p1[:-len(start)-1]
            vids.append((vid, start))
            if filename.endswith('.webm.part'):
                os.remove(os.path.join(dirname, filename))
        elif filename.endswith('.webm') or filename.endswith('.m4a') or filename.endswith('.temp.mp4'):
            p1 = filename.split('.')[0]
            try:
                vid, start = p1.split('_')
            except ValueError:
                start = p1.split('_')[-1]
                vid = p1[:-len(start) - 1]
            vids.append((vid, start))
            os.remove(os.path.join(dirname, filename))
        else:
            # check if the duration of the video stream is > a threshold in sec --> delete and re-download
            pr = subprocess.Popen(["bash", "check_with_ffprobe.sh", os.path.join(dirname, filename)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            std_out, std_err = pr.communicate()
            if eval(std_out):
                # should be deleted and re-downloaded later
                p1 = filename.split('.')[0]
                try:
                    vid, start = p1.split('_')
                except ValueError:
                    start = p1.split('_')[-1]
                    vid = p1[:-len(start) - 1]
                vids.append((vid, start))
                os.remove(os.path.join(dirname, filename))
rows = []
if len(vids) > 0:
    vids = list(set(vids))  # remove duplicates
    ids, st = zip(*vids)  # split video ids and start times
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] in ids and row[1] in st:
                rows.append(row)

# clear youtube-dl cache
pr = subprocess.Popen(["bash", "clear_ytdl_cache.sh"], stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, universal_newlines=True)
std_out, std_err = pr.communicate()

# resume download if last_index.json exists
if os.path.exists(os.path.join(args.data_path, 'last_index.json')):
    with open(os.path.join(args.data_path, 'last_index.json'), 'r') as f:
        data = json.load(f)
        last_idx = data['last_idx']
else:
    last_idx = 0

with open(csv_path, 'r') as f:
    try:
        if last_idx:
            reader = csv.reader(islice(f, last_idx, None))
        else:
            reader = csv.reader(f)

        # set up multiprocessing pool
        pool = mp.Pool(args.n_workers, initializer=initializer, maxtasksperchild=1)
        print(f"Setting a pool with a total of {args.n_workers} workers")
        # set up counter (to track how many videos were downloaded)
        counter = mp.Value('i', last_idx)
        # flag to retry download if necessary
        retry_flag = mp.Value('i', 0)

        for idx, row in enumerate(reader, last_idx):
            # prioritize videos that were not downloaded properly during the last run
            while len(rows) > 0:
                yt_id, st, _, dest = rows[0]
                rows.pop(0)
                dest = os.path.join(train_dir, '') if dest == 'train' else os.path.join(test_dir, '')
                pool.apply_async(download_snippet, [yt_id, st, dest, args.ffmpeg_path, args.cookies_path], callback=callback)

            yt_id, st, _, dest = row
            dest = os.path.join(train_dir, '') if dest == 'train' else os.path.join(test_dir, '')
            pool.apply_async(download_snippet, [yt_id, st, dest, args.ffmpeg_path, args.cookies_path], callback=callback)

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        exit()
    finally:
        # save last encountered index to json file (in the given data path)
        mode = 'r+' if os.path.exists(os.path.join(args.data_path, 'last_index.json')) else 'w'
        with open(os.path.join(args.data_path, 'last_index.json'), mode) as f2:
            if mode == 'w':
                temp_dict = dict(last_idx=counter.value)
                json.dump(temp_dict, f2)
            else:
                data = json.load(f2)
                data['last_idx'] = counter.value
                f2.seek(0)  # reset file pointer to position 0
                json.dump(data, f2)

