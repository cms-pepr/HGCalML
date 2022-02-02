
//only use bare c++ so that it compiled with nvcc
//template it to hack it header-only


//here T is a C-style int array, and n_dim is a specialised constant return function

//this is kept very simple on purpose



//to be removed
#include <iostream>

#include <initializer_list>

template<class T>
class index_vector {
public:

    index_vector(int i, int j, int k=0, int l=0, int m=0, int n=0, int o=0);

    index_vector<T> & operator += (const index_vector<T>& rhs){
        for(int i=0;i<n_dim();i++)
            indices_[i] += rhs.indices_[i];
        return *this;
    }
    index_vector<T> operator+(const index_vector<T>& rhs){
        index_vector<T> cp=*this;
        return cp+=rhs;
    }

    int operator[](int i)const{
        return indices_[i];
    }
    int& operator[](int i){
        return indices_[i];
    }

    int to_flat(const index_vector<T>& dims)const{
        int index=0;
        int mul = 1;
        for (int i = n_dim()-1; i != -1; i--) { //for (int  i = 0; i != n_dim(); ++i) {
            index += indices_[i] * mul;
            mul *= dims[i];
        }
        return index;
    }

    index_vector<T>& read_flat(int idx, const index_vector<T>& dims, int ntotal=-1){
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

    bool is_valid(const index_vector<T>& dims)const{//checks if indices fit in dimensions
        for(int i=0;i<n_dim();i++)
            if(dims.indices_[i]-indices_[i] <= 0)
                return false;
        return true;
    }
    int n_dim()const;


private:
    T indices_;

};


template<>
index_vector<int[2]>::index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j}{}
template<>
index_vector<int[3]>::index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k}{}
template<>
index_vector<int[4]>::index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l}{}
template<>
index_vector<int[5]>::index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l,m}{}
template<>
index_vector<int[6]>::index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l,m,n}{}
template<>
index_vector<int[7]>::index_vector(int i, int j, int k, int l, int m, int n, int o):indices_{i,j,k,l,m,n,o}{}

template<>
int index_vector<int[2]>::n_dim()const{
    return 2;
}
template<>
int index_vector<int[3]>::n_dim()const{
    return 3;
}
template<>
int index_vector<int[4]>::n_dim()const{
    return 4;
}
template<>
int index_vector<int[5]>::n_dim()const{
    return 5;
}
template<>
int index_vector<int[6]>::n_dim()const{
    return 6;
}
template<>
int index_vector<int[7]>::n_dim()const{
    return 7;
}


template<class T>
std::ostream &operator<<(std::ostream &s, index_vector<T> const &iv) {
    s << "[";
    for(int i=0;i<iv.n_dim();i++)
        s << iv[i] <<", ";
    s << "]";
    return s;
}

template<class T>
class binstepper {



};
