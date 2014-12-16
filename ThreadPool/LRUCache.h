#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_

typedef float Qfloat;
typedef signed char schar;

class LRUCache
{
public:
	virtual ~LRUCache() {}
	virtual int get_data(const int index, Qfloat **data, int len) = 0;
    virtual void swap_index(int i, int j) = 0;
};

#endif
