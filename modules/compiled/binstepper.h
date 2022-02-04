/*
 * only use bare c++ so that it compiled with nvcc
 * assert might need to be removed/exchanged
 * template only, some dirty hack.
 * for complaints: jkiesele
 */

//to be removed
#include <iostream>
#include <assert.h>

#include <cmath>
#include <initializer_list>


#include "cuda_runtime.h"

#define index_vector(x) _index_vector<int[x]>

template<class T>
class _index_vector {
public:

    _index_vector(){}

    _index_vector(int i, int j, int k=0, int l=0, int m=0, int n=0, int o=0);

    _index_vector<T> & operator += (const _index_vector<T>& rhs){
        for(int i=0;i<n_dim();i++)
            indices_[i] += rhs.indices_[i];
        return *this;
    }
    _index_vector<T> operator+(const _index_vector<T>& rhs){
        _index_vector<T> cp=*this;
        return cp+=rhs;
    }
    _index_vector<T> & operator -= (const _index_vector<T>& rhs){
        for(int i=0;i<n_dim();i++)
            indices_[i] -= rhs.indices_[i];
        return *this;
    }
    _index_vector<T> operator-(const _index_vector<T>& rhs){
        _index_vector<T> cp=*this;
        return cp-=rhs;
    }

    int operator[](int i)const{
        return indices_[i];
    }
    int& operator[](int i){
        return indices_[i];
    }

    void set_all(int idx){
        for(int i=0;i<n_dim();i++)
            indices_[i]=idx;
    }

    int to_flat(const _index_vector<T>& dims)const{
        int index=0;
        int mul = 1;
        for (int i = n_dim()-1; i != -1; i--) { //for (int  i = 0; i != n_dim(); ++i) {
            index += indices_[i] * mul;
            mul *= dims[i];
        }
        return index;
    }

    _index_vector<T>& read_flat(int idx, const _index_vector<T>& dims, int ntotal=-1){
        int mul = ntotal;
        if(mul<1){
            mul=1;
            for(int i=0;i<n_dim();i++)
                mul*=dims[i];
        }
        for (int  i = 1; i != n_dim()+1; ++i) { //for (int i = n_dim(); i != 0; --i) {
            mul /= dims[i - 1];
            indices_[i - 1] = idx / mul;
            idx -= indices_[i - 1] * mul;
        }
        return *this;
    }

    bool any_abs_bigger(int compare){
        for(int i=0;i<n_dim();i++){
            if(abs(indices_[i])>compare)
                return true;
        }
        return false;
    }
    bool at_least_one_abs_equals(int compare){
        for(int i=0;i<n_dim();i++){
            if(abs(indices_[i])==compare)
                return true;
        }
        return false;
    }


    _index_vector<T>& read_flat(int idx, int same_dims, int ntotal=-1){
        int mul = ntotal;
        if(mul<1){
            mul=1;
            for(int i=0;i<n_dim();i++)
                mul*=same_dims;
        }
        for (int  i = 1; i != n_dim()+1; ++i) { //for (int i = n_dim(); i != 0; --i) {
            mul /= same_dims;
            indices_[i - 1] = idx / mul;
            idx -= indices_[i - 1] * mul;
        }
        return *this;
    }

    bool is_valid(const _index_vector<T>& dims)const{//checks if indices fit in dimensions
        for(int i=0;i<n_dim();i++){
            if(dims.indices_[i]-indices_[i] <= 0)
                return false;
            if(indices_[i] < 0)
                return false;
        }
        return true;
    }

    bool is_valid(int same_dims)const{//checks if indices fit in dimensions
        for(int i=0;i<n_dim();i++){
            if(same_dims-indices_[i] <= 0)
                return false;
            if(indices_[i] < 0)
                return false;
        }
        return true;
    }

    int n_dim()const;

    template<class U>
    _index_vector<U> expand(int pos, int val)const{
        assert(pos<n_dim());
        _index_vector<U> out;
        assert(out.n_dim()>n_dim());
        int oi=0;
        for(int i=0;i<n_dim()+1;i++){
            out[i] = indices_[oi];
            if(pos!=i)
                oi++;
            else
                out[i]=val;
        }
        return out;
    }
    template<class U>
    _index_vector<U> squeeze(int pos)const{
        assert(pos<n_dim());
        _index_vector<U> out;
        int oi=0;
        for(int i=0;i<n_dim();i++){
            if(pos!=i){
                out[oi] = indices_[i];
                oi++;
            }
        }
        return out;
    }

    int multi()const{
        int mul=1;
        for(int i=0;i<n_dim();i++)
            mul*=indices_[i];
        return mul;
    }

private:
    T indices_;

};


template<>
_index_vector<int[2]>::_index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j}{}
template<>
_index_vector<int[3]>::_index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k}{}
template<>
_index_vector<int[4]>::_index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l}{}
template<>
_index_vector<int[5]>::_index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l,m}{}
template<>
_index_vector<int[6]>::_index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l,m,n}{}
template<>
_index_vector<int[7]>::_index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l,m,n,o}{}

template<>
int _index_vector<int[2]>::n_dim()const{
    return 2;
}
template<>
int _index_vector<int[3]>::n_dim()const{
    return 3;
}
template<>
int _index_vector<int[4]>::n_dim()const{
    return 4;
}
template<>
int _index_vector<int[5]>::n_dim()const{
    return 5;
}
template<>
int _index_vector<int[6]>::n_dim()const{
    return 6;
}
template<>
int _index_vector<int[7]>::n_dim()const{
    return 7;
}


//for debugging
template<class T>
std::ostream &operator<<(std::ostream &s, _index_vector<T> const &iv) {
    s << "[";
    for(int i=0;i<iv.n_dim();i++)
        s << iv[i] <<", ";
    s << "]";
    return s;
}

/*
 * don't include the row split here?
 *
 *
 * simplest 'continue' implementation for now. can be made smarter
 *
 * does support different binning in different dimensions in principle..
 *
 */

class binstepper_base {
public:
    virtual ~binstepper_base(){}

    virtual bool done()const=0;
    virtual int step(bool& valid)=0;
    virtual void set_distance(int d)=0;
};

#define binstepper(x) _binstepper<int[x]>

template<class T>//this is getting ridiculous, should be made prettier in next round
class _binstepper: public binstepper_base {
public:

    _binstepper(_index_vector<T> dims, int flat_offset):d_(1),dims_(dims){//start with 1 radius directly
        set_distance(1);
        offset_.read_flat(flat_offset, dims_);
    }

    void set_distance(int d){
        d_=d;
        visit_.set_all(2*d+1);//span cube
        d_offset_.set_all(-d);
        nsteps_ = std::pow(2*d_+1,visit_.n_dim());
        nsteps_left_ = nsteps_;
    }

    bool done()const{
        return nsteps_left_==0;
    }

    int step(bool& valid){
        if(done()){
            valid=false;
            return -100;
        }
        nsteps_left_--;

        // w.r.t. own capacity
        visit_.read_flat(nsteps_left_,2*d_+1,nsteps_);
        auto addidx = visit_ + d_offset_; //now these are index offsets

        if(!addidx.at_least_one_abs_equals(d_)){
            return step(valid);
        }

        _index_vector<T> out = offset_ + addidx;
        valid = out.is_valid(dims_);

        if(!valid)
            return step(valid);

        return out.to_flat(dims_);
    }


private:

    int nsteps_, nsteps_left_;
    int d_;
    _index_vector<T> dims_, visit_, d_offset_, offset_;

};











