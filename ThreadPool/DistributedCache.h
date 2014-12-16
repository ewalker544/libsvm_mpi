#ifndef _DISTRIBUTED_CACHE_POOL_H_
#define _DISTRIBUTED_CACHE_POOL_H_

#include <iostream>
#include <cstdlib>
#include <list>
#include <vector>
#include <functional>
#include <memory>
#include "mpi.h"

#include "SvmThreads.h"
#include "LRUCache.h"

#define INRANGE(x,y) (x == 0 && y == 2)

/**
 * Simple cache pool for allocating memory of fixed size
 */
class DistributedCache : public LRUCache
{
public:

	DistributedCache(int l_, const schar *y, std::function<double(int,int)> func) : l(l_)
	{
		setup(y, func);

		if (my_rank == 0) {
			// cache at least 50 columns or 10% if one node share
			max_local_cache = int(std::max(50.0, 0.1 * (l / world_size)));

			// table for all the other column indicies not in the local 
			int table_size = l - end;
			RemoteCacheTable.reserve(table_size);
			for (int i = 0; i < table_size; ++i) {
				RemoteCacheTable.push_back(new RemoteColumn());
			}
		}
	}

    virtual ~DistributedCache()
	{
		if (my_rank == 0) {
			shut_down();

			while ( !RemoteCacheTable.empty() ) 
			{
				RemoteColumn *ptr = RemoteCacheTable.back();
				RemoteCacheTable.pop_back();

				if (ptr->valid) {
					delete [] ptr->data;
				}

				delete ptr;
			}
		}

		delete [] pool; // free entire pre-allocated block
	}

   	/**
	 * Gets a column of height len from the cache pool 
	 * @return 0 if it is new memory that needs to be filled, or len if memory is already filled
	 */
    int get_data(const int index, Qfloat **data, int len) 
	{
		if (my_rank != 0) {
			std::cerr << "Rank " << my_rank << " should not be calling get_data()\n";
			MPI::COMM_WORLD.Abort(-1);
		}

		int partner_rank = find_rank(index);

		if (partner_rank != my_rank) {

			int tbl_idx = index - end;
			RemoteColumn *col = RemoteCacheTable[tbl_idx];

			if (col->valid) { // We have a hit!

				*data = col->data;
				CacheList.erase(col->pos); // remove from list

			} else {

				if (CacheList.size() >= max_local_cache) {
					RemoteColumn *ptr= CacheList.back(); // pick a column from the end of the list to reuse
					CacheList.pop_back();   // remove it from the list

					col->data = ptr->data;	// reuse its space
					col->valid = true;      // indicate this is now valid

					ptr->data = NULL;       // reset the old column
					ptr->valid = false;     // indicate its no longer valid

				} else {
					col->data = new Qfloat[l];
					col->valid = true;
				}	

				// Send request to partner rank for column data
				MPI::COMM_WORLD.Send(&index, 1, MPI_INT, partner_rank, 0);

				MPI::Status status;
				MPI::COMM_WORLD.Recv(col->data, l, MPI_FLOAT, partner_rank, 0, status);

				*data = col->data;
			}

			CacheList.push_front(col); // put this column in the front so we don't evict it
			col->pos = CacheList.begin(); // store its iterator so we can delete it if we have too

			return len;
		} else {
			int i_offset = index - start;
			*data = &pool[linear_index(i_offset, 0)];
			return len;
		} 
	}

    void swap_index(int i, int j)
	{
		std::cerr << "Fixed size cache pool cannot be used with shrinking!\n";
		MPI::COMM_WORLD.Abort(-1);
	}

private:

	static const int terminate_command = -1;

	int start, end;
	int my_rank, world_size;
	int l;
	long int size;
	size_t max_local_cache;

	Qfloat *stage[2];
	int next_pos;

	Qfloat *pool;

	int linear_index(int i, int j) 
	{
		return i * l + j;
	}

	void setup(const schar *y, std::function<double(int,int)> func)
	{
		SvmThreads * threads = SvmThreads::getInstance();

		world_size = MPI::COMM_WORLD.Get_size();
		my_rank = MPI::COMM_WORLD.Get_rank();

		start = my_rank * (l / world_size);
		end = ((my_rank == world_size-1) ? l : (my_rank + 1) * (l / world_size));

		size = l * (end-start);	 // size of my pool

		pool = new Qfloat[size];

		auto func1 = [&](int tid) {

			int pstart = start + threads->start(tid, end-start);
			int pend = start + threads->end(tid, end-start);

			for (int i = pstart; i < pend; ++i) {
				for (int j = 0; j < l; ++j) {
					int i_offset = i - start;
					pool[linear_index(i_offset, j)] = (Qfloat)(y[i]*y[j]*func(i, j));
				}
			}
		};

		threads->run_workers(func1);

		// Wait for all nodes to complete initializing their pool
		MPI::COMM_WORLD.Barrier();

		if (my_rank != 0) 
		{ 	
			// all ranks, other than 0, act as a cache server
			//
			// setup thread to respond to cache request
			auto func2 = [&] () {
				int index;
				MPI::Status status;

				while (true) {

					MPI::COMM_WORLD.Recv(&index, 1, MPI_INT, 0, 0, status);

					if (index >= start && index < end) {

						// serve cache values to node 0
						int i = index - start;
						MPI::COMM_WORLD.Send(&pool[linear_index(i, 0)], l, MPI_FLOAT, 0, 0);

					} 
					else if (index == terminate_command) {
						break;	
					} 
					else { 
						std::cerr << "Rank " << my_rank << " got invalid index " << index << ". Exiting ..." << std::endl;
						// index outside range.  exit loop.
						break;

					}
				}

			};

			std::thread server_thrd(func2);
			server_thrd.join();
		}
	}

	int find_rank(int index)
	{
		int group = l / world_size;
		int rank = index / group;
		return ((rank < world_size) ? rank : world_size-1);
	}

	void shut_down() {
		if (my_rank == 0) {
			int shut_down_msg = terminate_command;
			for (int i = 1; i < world_size; ++i) { // shuting down all nodes!
				MPI::COMM_WORLD.Send(&shut_down_msg, 1, MPI_INT, i, 0);
			}
		}
	}

	struct RemoteColumn {
		std::list<RemoteColumn *>::iterator pos; // position in list
		bool valid; // true if this is column is cached
		Qfloat *data; // Column data

		/** Constructor */
		RemoteColumn() {
			valid = false;
			data = NULL;
		}
	};

	std::vector<RemoteColumn *> RemoteCacheTable; // table for fast lookup

	std::list<RemoteColumn *> CacheList; // list of cached columns, limited to max_local_cache
};

#endif
