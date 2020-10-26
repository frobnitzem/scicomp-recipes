Multiple Implementations
########################

In C++, you can accomplish multiple coexisting implementations
for the same object in two ways -- the delegation pattern,
which uses templates, and the virtual extension pattern.

Delegation
----------

The delegation pattern populates a high-level
class with object-specific properties on creation.
That class then functions as an interface, delegating
all its operations to those functions::

    // main.cc
    #include "vector.hh"
    #include <iostream>

    int main(int argc, char *argv[]) {
        Vector x3 = mkHostVec();
        x3.print();
    }

The implementations for each instance of the object are created
through a series of functions that get passed into the object::

    // vector_host.cc and vector_cuda.cu
    #include <iostream>
    #include "vector.hh"

    void *initCUDA() {
        std::cout << "Device Ctor" << std::endl;
        return nullptr;
    }

    void dtorCUDA(void *data) {
        std::cout << "Device Dtor" << std::endl;
    }
    void printCUDA(void *data) {
        std::cout << "Device Vector" << std::endl;
    }
    Vector mkCUDAVec() {
        return Vector(initCUDA, dtorCUDA, printCUDA, Device::CUDA);
    }

    void *initHost() {
        std::cout << "Host Ctor" << std::endl;
        return nullptr;
    }
    void dtorHost(void *data) {
        std::cout << "Host Dtor" << std::endl;
    }
    void printHost(void *data) {
        std::cout << "Host Vector" << std::endl;
    }
    Vector mkHostVec() {
        return Vector(initHost, dtorHost, printHost, Device::Host);
    }

The high-level object that ties these together just
stores each function to use later::

    // vector.hh
    #include <functional>

    enum class Device { Host, CUDA };

    class Vector {
      private:
        std::function<void *()> _ctor;
        std::function<void(void *)> _dtor;
        std::function<void(void *)> _print;

      public:
        Device dev;

        Vector(std::function<void *()> &&ctor,
               std::function<void(void *)> &&dtor,
               std::function<void(void *)> &&print, Device _dev)
                : _ctor(ctor), _dtor(dtor), _print(print), dev(_dev) {
            data = _ctor();
        }
        ~Vector() {
            _dtor(data);
        }
        void print() {
            _print(data);
        }
        void *data;
    };

    Vector mkCUDAVec();
    Vector mkHostVec();


Virtual Extension
-----------------

High-level code can work entirely in terms of a base object::

    // main.cc
    #include "vector.hh"

    int main(int argc, char *argv[]) {
        Vector *x3 = mkHostVec();
        x3->print();
        delete x3;
    }

The disadvantage of this apprach is that it requires
manual memory management with pointers.

The implementation codes can be separately named classes, or
template specializations -- either one::

    // vector_host.cc and vector_cuda.cu
    #include "vector.hh"
    #include <iostream>

    template<>
    void DevVector<Device::Host>::print() {
        std::cout << "Host Vec" << std::endl;
    }
    template<>
    DevVector<Device::Host>::DevVector() {
            std::cout << "Ctor Host" << std::endl;
    }
    template<>
    DevVector<Device::Host>::~DevVector() {
            std::cout << "Dtor Host" << std::endl;
    }
    Vector *mkHostVec() {
        return new DevVector<Device::Host>();
    }

    template<>
    void DevVector<Device::CUDA>::print() {
        std::cout << "Device Vector" << std::endl;
    }
    template<>
    DevVector<Device::CUDA>::DevVector() {
        std::cout << "Ctor Device" << std::endl;
    }
    template<>
    DevVector<Device::CUDA>::~DevVector() {
        std::cout << "Dtor Device" << std::endl;
    }
    Vector *mkCUDAVec() {
        return new DevVector<Device::CUDA>();
    }

The ``vector.hh`` header file has all the interesting parts::

    // Declare tags for host and CUDA spaces
    enum class Device { Host, CUDA };

    // The base vector class defines only prototypes
    struct Vector {
        virtual void print() = 0;
        virtual ~Vector() {};
    };

    // These classes implement the Vector interface.
    template <Device dev>
    struct DevVector : public Vector {
        void print();
        DevVector();
        ~DevVector();
        void *data;
    };

    Vector *mkHostVec();
    Vector *mkCUDAVec();

.. note::

    Contributed by David M. Rogers

