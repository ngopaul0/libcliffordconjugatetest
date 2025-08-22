#ifndef COMPLEXEXACTREPR_H
#define COMPLEXEXACTREPR_H

#include <complex>

struct ComplexExactRepr {
    // union allows access to same memory location using different types
    union {
        double d;
        long long i;
    } realValue{};

    union {
        double d;
        long long i;
    } imagValue{};

    explicit ComplexExactRepr(const std::complex<double> d) {
        realValue.d = d.real();
        imagValue.d = d.imag();
    }

    ComplexExactRepr(const long long realRepr, const long long imagRepr) {
        realValue.i = realRepr;
        imagValue.i = imagRepr;
    }

    [[nodiscard]] std::complex<double> cmplx() const { return {realValue.d, imagValue.d}; }

    friend std::ostream& operator<<(std::ostream& os, const ComplexExactRepr& obj) {
        // os << "{" << obj.realValue.i << "," << obj.imagValue.i << "}";
        // Allows for printing out the exact representation of the complex number so that it
        // can be reconstructed exactly. This prints out a statement that can be pasted in C++
        // to recreate the *exact* complex number as it is in memory.
        os << "ComplexExactRepr(" << obj.realValue.i << "," << obj.imagValue.i << ").cmplx()";
        return os;
    }
};

#endif // COMPLEXEXACTREPR_H
