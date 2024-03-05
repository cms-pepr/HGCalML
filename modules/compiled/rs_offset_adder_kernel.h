
#ifndef RS_OFFSET_ADDER_KERNEL_H
#define RS_OFFSET_ADDER_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct RSOffsetAdderOpFunctor {
    void operator()(
            const Device &d,
            const int * t_dx,
            const int * rs,
            int * new_t_idx,

            const int n_vert,
            const int n_rs
    );

};

}  // namespace functor
}  // namespace tensorflow


#endif //RS_OFFSET_ADDER_KERNEL_H
