# L3 — caller-owned scratch. A leaf despite its index: a bump allocator needs
# neither parameters nor tables. The index orders what may depend on it, not
# what it depends on.
fhn_ckks_layer(fhn_ckks_arena LAYER 3 SOURCES Arena.cpp)
