from constant import ACL_ERROR_NONE

def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))
    # else:
    #     print("{} success".format(message))