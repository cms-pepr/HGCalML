
#ifndef INDEX_REPLACER_KERNEL_H
#define INDEX_REPLACER_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct IndexReplacerOpFunctor {
    void operator()(
            const Device &d,
            const int * to_be_replaced,
            const int * replacements,
            int * replaced,

            const int n_to_be_replaced,
            const int n_replacements
    );

};

}  // namespace functor
}  // namespace tensorflow


#endif //INDEX_REPLACER_KERNEL_H
