import multiprocessing as mp
import os
import random
import time

import cv2
import gym
import minerl
import psutil
from easydict import EasyDict
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from grl.utils import set_seed

image_size = 64
config = EasyDict(
    dict(
        data=dict(
            image_size=image_size,
            data_path="./minerl_images",
            origin_image_size=(360, 640, 3),
        ),
    )
)


def kill_process_by_string(process_string):
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if type(proc.info["cmdline"]) is list:
            if process_string in " ".join(proc.info["cmdline"]):
                print(
                    f"Killing process {proc.info['pid']}: {proc.info['name']} - {' '.join(proc.info['cmdline'])}"
                )
                proc.kill()


def resize_image(image, size):
    # image is numpy array of shape (360, 640, 3), resize to (64, 64, 3)
    return cv2.resize(image[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)
    # return cv2.resize(image[:, :, ::-1], (128, 72), interpolation=cv2.INTER_AREA)
    # return cv2.resize(image[:, :, ::-1], (128, 72))


def save_image(image, path):
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def ramdon_action(method, env):
    if method == "random":
        action = env.action_space.sample()
        action["camera"] = [random.uniform(-1, 1), random.uniform(-2, 2)]
        action["ESC"] = 0
        action["inventory"] = 0
        action["drop"] = 0
        action["hotbar.1"] = 0
        action["hotbar.2"] = 0
        action["hotbar.3"] = 0
        action["hotbar.4"] = 0
        action["hotbar.5"] = 0
        action["hotbar.6"] = 0
        action["hotbar.7"] = 0
        action["hotbar.8"] = 0
        action["hotbar.9"] = 0
        if random.uniform(0, 1) < 0.9:
            action["back"] = 0
        return action
    elif method == "run_forward":
        action = env.action_space.sample()
        action["camera"] = [random.uniform(-1, 1), random.uniform(-2, 2)]
        action["attack"] = 0
        action["ESC"] = 0
        action["inventory"] = 0
        action["drop"] = 0
        action["hotbar.1"] = 0
        action["hotbar.2"] = 0
        action["hotbar.3"] = 0
        action["hotbar.4"] = 0
        action["hotbar.5"] = 0
        action["hotbar.6"] = 0
        action["hotbar.7"] = 0
        action["hotbar.8"] = 0
        action["hotbar.9"] = 0
        action["back"] = 0
        action["sprint"] = 1
        return action
    else:
        raise NotImplementedError(f"Unknown method {method}")


def collect_data(env_id, i, data_path):
    try:
        env = gym.make(env_id)
        obs = env.reset()
        done = False
        counter = 0
        while not done and counter < 100:
            action = ramdon_action("run_forward", env)
            obs, reward, done, info = env.step(action)

            save_image(
                resize_image(obs["pov"], config.data.image_size),
                os.path.join(data_path, f"{env_id}_{i}_{counter}.png"),
            )

            # env.render()
            counter += 1
        env.close()
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    set_seed()
    mp.set_start_method("spawn")

    if not os.path.exists(config.data.data_path):
        os.makedirs(config.data.data_path)

    for env_id in [
        "MineRLBasaltFindCave-v0",
        "MineRLBasaltCreateVillageAnimalPen-v0",
        "MineRLBasaltMakeWaterfall-v0",
        "MineRLBasaltBuildVillageHouse-v0",
        "MineRLObtainDiamondShovel-v0",
    ]:

        i = 0
        while i < 1000:
            p = mp.Process(target=collect_data, args=(env_id, i, config.data.data_path))
            try:
                print(f"Starting process {env_id}-{i}")
                p.start()
                p.join()
                print(f"Process {env_id}-{i} finished")
                i += 1
            except Exception as e:
                print(e)
                p.terminate()
                kill_process_by_string("launchClient.sh")
                kill_process_by_string("build/libs/mcprec-6.13.jar")
                time.sleep(10)
                continue
