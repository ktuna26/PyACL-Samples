"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2020-6-04 20:12:13
MODIFIED: 2021-10-31 23:48:45
"""

# -*- coding:utf-8 -*-
from constant import ACL_ERROR_NONE


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))
    # else:
    #     print("{} success".format(message))