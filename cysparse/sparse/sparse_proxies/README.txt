See #113 for more details.

In short: DO NOT remove p_mat.pxd_OLD and p_mat.pyx_OLD: this is a base class to define a proxy. For whatever reason,
I (Nikolaj) didn't manage to make this design work.

Temporary solution (that might be definitive): duplicate the code of this base class in ALL proxies.

(By the way: I lost more than 10 hours on this...).