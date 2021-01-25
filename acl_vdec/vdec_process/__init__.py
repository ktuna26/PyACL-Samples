import acl
from .acl_util import check_ret

ret = acl.init()
check_ret("acl.init", ret)