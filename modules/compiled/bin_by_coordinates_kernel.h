
#ifndef BIN_BY_COORDINATES_KERNEL_H
#define BIN_BY_COORDINATES_KERNEL_H

namespace tensorflow {
namespace functor {

template<typename Device, typename dummy>
struct BinByCoordinatesOpFunctor {
    void operator()(
            const Device &d
    );



};

}  // namespace functor
}  // namespace tensorflow

#endif //BIN_BY_COORDINATES_KERNEL_H
