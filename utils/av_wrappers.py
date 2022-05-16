import av
import numpy as np
from fractions import Fraction


def av_open(inpt):
    return av.open(inpt)


def av_load_video(container, video_fps=None, duration=None, start_time=0.0):
    """
    Load video stream
    :param container: Input container
    :param video_fps: Target fps at which frames are extracted (default: None)
    :param duration: Duration of the extracted stream (default: None, i.e. the whole stream is returned)
    :param start_time: Timestamp of first frame in the stream (default: 0)
    :return: list of video frames (PIL images), video fps, start time
    """

    video_stream = container.streams.video[0]
    _ss = video_stream.start_time * video_stream.time_base  # should be zero in our case!
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur
    _fps = video_stream.average_rate

    if video_fps is None:
        video_fps = _fps

    if duration is None:
        duration = _ff - start_time

    # figure out which frames should be decoded
    output_times = np.arange(start_time, min(start_time + duration - 0.5 / _fps, _ff), 1. / video_fps)[:int(duration * video_fps)].tolist()
    output_frames = [int(_fps * (t - _ss)) for t in output_times]
    start_time = output_frames[0] / float(_fps)

    # seek the nearest frame to the start timestamp
    container.seek(int(start_time * av.time_base))  # offset is in av.time_base here

    # decode snippet
    frames = []
    for frame in container.decode(video=0):
        if len(frames) == len(output_frames):
            break  # all frames have been decoded
        frame_no = frame.pts * frame.time_base * _fps  # number of frame = frame's presentation timestamp (in sec) * fps
        if frame_no < output_frames[len(frames)]:
            continue  # not the frame we want

        # decode frame
        pil_img = frame.to_image()  # get an RGB PIL.Image of the frame
        while frame_no >= output_frames[len(frames)]:
            frames += [pil_img]
            if len(frames) == len(output_frames):
                break  # all frames have been decoded

    return frames, video_fps, start_time


def av_load_audio(container, audio_srate=None, duration=None, start_time=0.0):
    """
    Load audio stream
    :param container: Input container
    :param audio_srate: Target audio sampling rate at which the stream will be extracted
    :param duration: Duration of the extracted stream (default: None, i.e. the whole stream is returned)
    :param start_time: Timestamp of first frame in the stream (default: 0)
    :return: numpy.ndarray of shape (1 x L) i.e. the length of the audio stream, audio sampling rate
    """

    try:
        audio_stream = container.streams.audio[0]
    except IndexError:
        # audio stream does not exist!
        if duration is None:
            # In this case, all videos are 10 seconds long (approximately).
            # Instead of using a fixed value, we could infer total duration from the video stream, i.e.
            # duration = container.streams.video[0].duration * container.streams.video[0].time_base
            output = np.zeros((1, int(audio_srate * 10.)))
        else:
            output = np.zeros((1, int(audio_srate * duration)))
        return output, audio_srate
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _rate = audio_stream.sample_rate

    if audio_srate is None:
        resample = False
        audio_srate = _rate
    else:
        resample = True
        audio_resampler = av.audio.resampler.AudioResampler(format="s16p", layout="mono", rate=audio_srate)

    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    # seek the nearest frame to the start timestamp
    container.seek(int(start_time * av.time_base))

    # decode snippet
    data, timestamps = [], []
    for frame in container.decode(audio=0):
        frame_pts = frame.pts * frame.time_base
        # end of frame = frame's presentation time stamp (in sec) + samples' duration (in sec)
        frame_end_pts = frame_pts + Fraction(frame.samples, frame.sample_rate)
        if frame_end_pts < start_time:
            continue  # clip has not started yet
        if frame_pts > end_time:
            break  # clip has been extracted

        try:
            frame.pts = None
            if resample:
                np_snd = audio_resampler.resample(frame).to_ndarray()
            else:
                np_snd = frame.to_ndarray()
            data += [np_snd]
            timestamps += [frame_pts]
        except AttributeError:
            continue  # don't use break here (could yield empty data). Instead, continue decoding and pad afterwards.

    try:
        data = np.concatenate(data, 1)
    except ValueError:
        if duration is None:
            output = np.zeros((1, int(audio_srate * 10.)))
        else:
            output = np.zeros((1, int(audio_srate * duration)))
        return output, audio_srate

    # decode audio
    decoding_start_time = timestamps[0]
    ss = int(audio_srate * (start_time - decoding_start_time))
    t = int(duration * audio_srate)
    if ss < 0:
        # beginning of the decoded sequence is after the predefined start time
        data = np.pad(data, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    if t > data.shape[1]:
        # the decoded sequence lasts less than the predefined duration
        data = np.pad(data, ((0, 0), (0, t - data.shape[1])), 'constant', constant_values=0)

    data = data[:, ss:ss + t]
    data = data / np.iinfo(data.dtype).max

    if data.shape[1] < t:
        # make sure that all waveforms have the same shape
        data = np.pad(data, ((0, 0), (0, t - data.shape[1])), 'constant', constant_values=0)

    return data, audio_srate

