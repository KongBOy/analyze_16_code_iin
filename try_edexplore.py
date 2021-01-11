### 嘗試把 edexplore 寫出來
# import os,inspect,edflow
# print("here:", os.path.dirname(inspect.getfile(edflow)))  ###   C:\Users\TKU\Anaconda3\envs\pytorch\Lib\site-packages\edflow
### 意思應該就是要定位出 edflow裝哪裡，然後進去執行 explore.py這樣子，下面乾脆直接把 explore.py複製過來囉！

###########################################################################
import pdb
### 直接把 explore.py 複製過來
import os
import sys
import random
import argparse
import yaml
import streamlit as st
from edflow.config import parse_unknown_args, update_config

import numpy as np
from edflow.util import walk, pp2mkdtable
from edflow import get_obj_from_str


def isimage(obj):
    return (
        isinstance(obj, np.ndarray)
        and len(obj.shape) == 3
        and obj.shape[2] in [1, 3, 4]
    )


def isflow(obj):
    return isinstance(obj, np.ndarray) and len(obj.shape) == 3 and obj.shape[2] in [2]


def istext(obj):
    return isinstance(obj, (int, float, str, np.integer, np.float))


def display_default(obj):
    if isimage(obj):
        return "Image"
    elif istext(obj):
        return "Text"
    elif isflow(obj):
        return "Flow"
    else:
        return "None"


def display(key, obj):
    st.subheader(key)
    sel = selector(key, obj)
    if sel == "Text":
        st.text(obj)

    elif sel == "Image":
        st.image((obj + 1.0) / 2.0)

    elif sel == "Flow":
        import flowiz as fz

        img = fz.convert_from_flow(obj)
        st.image(img)


def selector(key, obj):
    options = ["Auto", "Text", "Image", "Flow", "None"]
    idx = options.index(display_default(obj))
    select = st.selectbox("Display {} as".format(key), options, index=idx)
    return select


def show_example(dset, idx):
    ex = dset[idx]
    st.header("Keys")
    walk(ex, display, pass_key=True)
    st.header("Summary")
    summary = pp2mkdtable(ex, jupyter_style=True)
    # print markdown summary on console for easy copy and pasting in readme etc
    print(summary)
    st.markdown(summary)


def _get_state(config):
    Dataset = get_obj_from_str(config["dataset"])
    dataset = Dataset(config)
    return dataset


def explore(config, disable_cache=False):
    ### 走到這裡 disable_cache 是 False
    if not disable_cache:  ### 所以走這個
        get_state = st.cache(persist=False, allow_output_mutation=True)(_get_state)
    else:
        get_state = _get_state
    dset = get_state(config)
    dset.expand = True
    st.title("Dataset Explorer: {}".format(type(dset).__name__))

    input_method = st.sidebar.selectbox(
        "Index selection method", ["Slider", "Number input", "Sample"]
    )
    if input_method == "Slider":
        idx = st.sidebar.slider("Index", 0, len(dset), 0)
    elif input_method == "Number input":
        idx = st.sidebar.number_input("Index", 0, len(dset), 0)
    elif input_method == "Sample":
        idx = 0
        if st.sidebar.button("Sample"):
            idx = np.random.choice(len(dset))
        st.sidebar.text("Index: {}".format(idx))

    show_example(dset, idx)

    st.header("config")
    cfg_string = pp2mkdtable(config, jupyter_style=True)
    cfg = st.markdown(cfg_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="edflow dataset explorer")
    parser.add_argument(
        "-d",
        "--disable_cache",
        action="store_true",
        help="Disable caching dataset instantiation.",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="config.yaml",
        help="Paths to base configs.",
        default=list(),
    )
    try:
        sys.path.append(os.getcwd())
        opt, unknown = parser.parse_known_args()
        additional_kwargs = parse_unknown_args(unknown)
        config = dict()
        for base_config in opt.base:
            with open(base_config) as f:
                config.update(yaml.full_load(f))
        update_config(config, additional_kwargs)
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)
    explore(config, disable_cache=opt.disable_cache)

    ### 在終端機打的指令： python try_edexplore.py --dataset iin.data.<dataset選擇如下>
    ###     MNISTTrain, MNISTTest, FashionMNISTTrain, FashionMNISTTest, CIFAR10Train, CIFAR10Test, CelebATrain, CelebATest, FactorCelebATrain, FactorCelebATest, ColorfulMNISTTrain, ColorfulMNISTTest, SingleColorfulMNISTTrain, SingleColorfulMNISTTest
    ###     譬如：
    ###         python try_edexplore.py --dataset iin.data.MNISTTrain
    ###         python try_edexplore.py --dataset iin.data.MNISTTest
    ###         python try_edexplore.py --dataset iin.data.FashionMNISTTrain
    ###         python try_edexplore.py --dataset iin.data.FashionMNISTTest
    ###         python try_edexplore.py --dataset iin.data.CIFAR10Train
    ###         python try_edexplore.py --dataset iin.data.CIFAR10Test
    ###         python try_edexplore.py --dataset iin.data.CelebATrain
    ###         python try_edexplore.py --dataset iin.data.CelebATest
    ###         python try_edexplore.py --dataset iin.data.FactorCelebATrain
    ###         python try_edexplore.py --dataset iin.data.FactorCelebATest
    ###         python try_edexplore.py --dataset iin.data.ColorfulMNISTTrain
    ###         python try_edexplore.py --dataset iin.data.ColorfulMNISTTest
    ###         python try_edexplore.py --dataset iin.data.SingleColorfulMNISTTrain
    ###         python try_edexplore.py --dataset iin.data.SingleColorfulMNISTTest
    ###
    ###         東西會下載到 C:\Users\TKU/.eddata_cache\FashionMNIS


def fun(a, b, c, d):
    pass
