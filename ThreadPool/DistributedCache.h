#ifndef _DISTRIBUTED_CACHE_POOL_H_
#define _DISTRIBUTED_CACHE_POOL_H_

#include <iostream>
#include <cstdlib>
#include <list>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <limits>
#include <memory>
#include <mpi.h>

#include "SvmThreads.h"
#include "LRUCache.h"


/**
 * Simple cache pool for allocating memory of fixed size
 */
class DistributedCache : public LRUCache
{
public:

	DistributedCache(int l_, const schar *y_, std::function<double(int,int)> func_) : 
		l(l_), kernel(func_)
	{
		y = new schar[l];
		memcpy(y, y_, l * sizeof(schar));

		setup();
	}

    virtual ~DistributedCache()
	{
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

		delete [] y;

	}

   	/**
	 * Gets index column - locally or from a remote node
	 * @return 		0 if it is new memory that needs to be filled, 
	 * 				or len if memory is already filled
	 */
    int get_data(const int index, Qfloat **data, int len) 
	{
		Qfloat *d = get_column(index, false); // check if column is in local cache

		if (d == NULL) { // column not in cache

			int partner_rank = find_rank(index); // find out where it is located

			if (partner_rank == 0) {  
				// generate and cache column in rank 0

				*data = get_column(index, true);
				return len;

			} else {
				// Send request to partner rank for column data

				MPI::COMM_WORLD.Send(&index, 1, MPI_INT, partner_rank, 0);

				d = get_column_cache(index); // can a space in the cache to fill in the column data

				MPI::Status status;
				MPI::COMM_WORLD.Recv(d, l, MPI_FLOAT, partner_rank, 0, status);

				*data = d;
				return len;
			}
		}
		else { // found column data in cache
			*data = d;
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
	schar *y;
	std::function<double(int,int)> kernel;

	long int size;
	size_t max_local_cache;

	Qfloat *stage[2];
	int next_pos;

	int linear_index(int i, int j) 
	{
		return i * l + j;
	}

	void setup()
	{

		world_size = MPI::COMM_WORLD.Get_size();
		my_rank = MPI::COMM_WORLD.Get_rank();

		start = my_rank * (l / world_size);
		end = ((my_rank == world_size-1) ? l : (my_rank + 1) * (l / world_size));

		// table for all the other column indicies maintained by local cache
		int table_size;
		if (my_rank == 0)
			table_size = l; // master rank keeps track of everything
		else
			table_size = end - start; // slaves only keep track of their share

		RemoteCacheTable.reserve(table_size);
		for (int i = 0; i < table_size; ++i) {
			RemoteCacheTable.push_back(new RemoteColumn());
		}

		size_t allocated_memory = table_size * sizeof(RemoteColumn);

		size_t max_size = (getPhysicalMemory() - allocated_memory) / sizeof(Qfloat); // max # Qfloats
		size_t max_columns = max_size / l; // each column of size l

		double ratio = 0.8; // try caching 80% of the share of columns for each node

		// make sure we are not storing more than 95% of the available physical memory
		max_local_cache = size_t(std::min(ratio * (l / world_size), 0.9 * static_cast<double>(max_columns)));

		std::cerr << my_rank << ": caching " << max_local_cache << " columns\n";

		// Wait for all nodes to synchronize
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
						Qfloat *data = get_column(index, true);
						MPI::COMM_WORLD.Send(data, l, MPI_FLOAT, 0, 0);

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

	size_t getPhysicalMemory()
	{
		size_t default_rc = (2000 << 20); // 2 GB
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
		return (size_t)sysconf( _SC_PHYS_PAGES ) * (size_t)sysconf( _SC_PAGESIZE );
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
		return (size_t)sysconf( _SC_PHYS_PAGES ) * (size_t)sysconf( _SC_PAGE_SIZE );
#else
		std::ifstream fin("/proc/meminfo");

		if (fin.fail()) {
			return default_rc;
		}

		std::string token;
		while(fin >> token) {
			if(token == "MemTotal:") {
				size_t mem;
				if(fin >> mem) {
					return mem * 1024;
				} else {
					return default_rc;
				}
			}
			// ignore rest of the line
			fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		return default_rc;
#endif
	}

	Qfloat *get_column(int index, bool fill) 
	{
		int tbl_idx;

		if (my_rank == 0) {
			tbl_idx = index; // master is keeping track of everything
		} else {
			tbl_idx = index - start;
		}

		RemoteColumn *col = RemoteCacheTable[tbl_idx];

		if (col->valid) { // hit!

			CacheList.erase(col->pos); // remove from list

		} else { // miss!

			if (!fill)
				return NULL;

			if (CacheList.size() >= max_local_cache) {
				RemoteColumn *ptr= CacheList.back(); // pick a column from the end of the list to reuse
				CacheList.pop_back();   // remove it from the list

				col->data = ptr->data;	// reuse its space
				fill_column(index, col->data);
				col->valid = true;      // indicate this is now valid

				ptr->data = NULL;       // reset the old column
				ptr->valid = false;     // indicate its no longer valid

			} else {
				col->data = new Qfloat[l];
				fill_column(index, col->data);
				col->valid = true;
			}	
		}

		CacheList.push_front(col); // put this column in the front so we don't evict it
		col->pos = CacheList.begin(); // store its iterator so we can delete it later

		return col->data;
	}

	void fill_column(int i, Qfloat *data)
	{
		SvmThreads * threads = SvmThreads::getInstance();

		auto func = [&](int tid) {

			int start = threads->start(tid, l);
			int end = threads->end(tid, l);

			for (int j = start; j < end; ++j) {
				data[j] = (Qfloat)(y[i]*y[j]*kernel(i, j));
			}
		};

		threads->run_workers(func);

		return ;
	}

	Qfloat * get_column_cache(int index)
	{
		RemoteColumn *col = RemoteCacheTable[index];

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

		CacheList.push_front(col); // put this column in the front so we don't evict it
		col->pos = CacheList.begin(); // store its iterator so we can delete it later

		return col->data;
	}
};

#endif
