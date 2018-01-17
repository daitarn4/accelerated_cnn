/*
 * FPGA_channels.hpp
 *
 *  Created on: Apr 16, 2015
 *      Author: root
 */

#ifndef FPGA_CHANNELS_H
#define FPGA_CHANNELS_H

#ifdef ALTERA_CL
#define CHANNEL(TYPE,NAME,DEPTH) channel TYPE NAME;
#define CHAN(TYPE,DEPTH) channel TYPE
#else
#define CHAN(TYPE,DEPTH) NallaChannel<TYPE,DEPTH>
#define CHANNEL(TYPE,NAME,DEPTH) NallaChannel<TYPE,DEPTH> NAME;

#include <deque>
#ifdef THREAD_SAFE
#include <pthread.h>
#endif
#ifndef LINUX
#include "Windows.h"
#define usleep Sleep
#endif

template <typename Stype,int depth>
class NallaChannel
{
private:
	bool locked;
	int mdepth;
public:
	std::deque<Stype> local_queue;
	int count;
#ifdef THREAD_SAFE
	pthread_mutex_t mut;
#endif

public :
	NallaChannel():locked(false)
	{
		mdepth = depth;count=0;
#ifdef THREAD_SAFE
		mut = PTHREAD_MUTEX_INITIALIZER;
#endif
	}
	~NallaChannel(){}
	void Lock()
	{
		//while (locked)
		//	usleep(1);
		//locked = true;
#ifdef THREAD_SAFE
		pthread_mutex_lock(&mut);
#endif
	}
	void UnLock()
	{
#ifdef TRHEAD_SAFE
		pthread_mutex_unlock(&mut);
#endif
		//locked = false;
	}
};

template <typename Stype,int depth>
inline Stype read_channel_intel(NallaChannel<Stype,depth>  &q)
{
	//while (q.local_queue.size() < 1) //
//		usleep(1);
	q.Lock();
	Stype temp = q.local_queue.back();
	q.local_queue.pop_back();
	q.count--;
	q.UnLock();
	return temp;
}

template <typename Stype,int depth>
inline void write_channel_intel(NallaChannel<Stype,depth>  &q,Stype data)
{
	//while (q.count >= depth) //
	//	usleep(1);
	q.Lock();
	q.local_queue.push_front(data);
	q.count++;
	q.UnLock();
}
#endif
#endif /* FPGA_CHANNELS_H */
