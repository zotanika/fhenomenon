FHENOMENON
---
## **1. Introduction**
Fully Homomorphic Encryption (FHE) enables computations on encrypted data without decrypting it first, thus preserving data privacy during processing. Despite its powerful potential, FHE is complex and resource-intensive, making it challenging for widespread adoption. **Fhenomenon** is a C++ library designed to bridge this gap, providing a flexible, user-friendly interface that abstracts the complexities of FHE. This white paper outlines the architecture, design principles, and implementation details of **Fhenomenon**, highlighting how it simplifies FHE use while supporting a wide range of backend libraries and acceleration hardware.

## **2. Objectives**
The primary objectives of **Fhenomenon** are:
- **Ease of Use**: Avoid cryptography-native APIs or terminologies, provide more intuitive APIs minimize the need for users to understand the underlying theoretical details of FHE.
- **Flexibility**: Support multiple FHE libraries as backends through a common interface, enabling dynamic linking of external FHE libraries as plugins.
- **Performance**: Leverage hardware acceleration and optimized scheduling to enhance the efficiency of FHE operations.
- **Security**: Ensure robust key management.

## 3. Requirements
- The implementation of FHE cryptographic primitives can be outsourced to external libraries as well as **Fhenomenon**'s built-in implementation. Such actual FHE implementations are considered as `Backends` in **Fhenomenon**. Accordingly, the remaining parts that are not directly related to homomorphic encryption implementation can be referred to as `Frontend`.
- Under assuming a common interface exists among various open-sourced FHE libraries, external libraries can be dynamically linked as plugins. In reality, the FHE ecosystem is not yet technologically mature enough to provide a standardized interface as a software product, so full support may be challenging for now. Of course, the built-in implementations are also based on this common interface and have an acceleration layer that delegates computational workloads to hardware accelerators.
- When using this library, the user should be able to write code with minimal knowledge. For example, in `Compuon<T>`, `T` can be any basic C++ type such as `int`, `float`, `double`, or `complex`. By specifying the mathematical properties of the set that `Compuon<T>` should belong to at a desired point, `Compuon<T>` is mapped as an element of the set defined by the given properties. According to the mathematical context of FHE, mapping an array of plaintext data into a quotient ring polynomial form is called encryption.
- For example, the code snippet `Compuon<int> a = 10; a.belong(rlwe::ckks::configA); a.belong(rlwe::bgv::configA);` means the programmer declares a message of an integer scalar 10, encrypts it with CKKS configA parameters (automatically padding remaining slots with 0), and then transcrypts the ciphertext with BGV configA parameters. `T` can be a plain C POD struct, and if using RLWE-based schemes (e.g., CKKS), each element of the struct can be compiled to map to each slot of a CKKS message. In other words, the user should not need to manually create their own class by inheriting the basic `Compuon<T>` class.
- It supports a runtime detector that detects installed acceleration hardware in the system.
- A scheduler consisting of several modules is also required. For example:
    - `Receiver` collects frontend API calls from the user.
    - `Scheduler` constructs a complete graph representing the relationships between operations and ciphertext nodes.
    - `Strategy` provides a list of strategies to modify and optimize the scheduling graph in a manner similar to compiler passes, and serves as a framework that allows open-source community developers to add techniques to optimize homomorphic encryption operations, based on basic templates.
    - `Dispatcher` dispatches computational workloads to appropriate devices, such as dispatching some subgraphs to specific accelerators (GPUs) and the rest to the CPU.
    - One of the purposes of tracking operations in the scheduler is to facilitate defining coarse-grained parallel processing kernels, by fusing operations applied to a single ciphertext or bundling multiple ciphertexts involved in common operations, maximizing data reuse and minimizing data I/O by keeping loaded data in accelerators as much as possible. To achieve this, `Receiver` and `Scheduler` should build a graph that easily recognizes computational partitions per ciphertext object, accumulate operations along the partitions, and manage them based on the lifecycle of the ciphertext object until it expires.
    - This functionality does not need to be as complex as a real compiler for programming languages, but a well-structured framework is needed to help developers add manually scheduled strategies according to well-defined templates (e.g., lambda functions) and generate bundles of computations to be dispatched to the backend.
- The `BuiltinBackend` itself has an extended API list closely related to FHE libraries, unlike the basic frontend API that excludes cryptographic concepts as much as possible. This is similar to the general APIs typically provided by open-source FHE libraries (which `ExternalBackend` inherits and provides as is). The `BuiltinBackend` interface is an extended version of these basic APIs, with a much more extensive catalog of combinations to maximize parallel processing, such as fusing multiple operations and batching multiple ciphertexts. Naturally, some acceleration hardware may only partially support or not support these extended APIs at all; highly programmable GPUs may support all these extensions. Thus, the `BuiltinBackend` needs a delegate pattern to keep the hardware binding development process of the acceleration kernel independent of the `BuiltinBackend` development itself.
- `ExternalBackend` receives the shared library file path from the user during object creation.
- The frontend programming model should provide an intuitive interface for users. For instance, a Pythonic interface like `Session sess; { code block need to be scheduled }.run()` would be desirable. If the user calls an API outside the scope, it executes immediately; if called inside the scope, the code block is executed in a scheduling/optimization manner.
- Just `.run()` should guarantee execution on the actual acceleration hardware. The user first decides whether to use the built-in or external library when creating the Backend object, which is then brought in as an argument to the `Session`.
- The `Backend` should be designed as a singleton pattern because it represents the FHE system, making it intuitive for the user to perceive that the system environment is bound to FHE as a building block. Since the optimization strategy is determined by the type of Backend, the `Scheduler` also becomes a substructure of the `Backend`. Specifically, the `Scheduler` related methods in `Backend` are `virtual` and are concretely overridden in `BuiltinBackend` or `ExternalBackend`.
- Setting scheme parameters directly can be quite cumbersome for the user, so the scheduler should determine the parameters for the code blocks written within the session scope, with this process hidden from the user.
- Let's consider key management. In FHE schemes, the secret key must be kept hidden, and the public encryption key and evaluation keys are needed to preserve the algebraic structure of the distorted ciphertext through homomorphic operations. These public keys may be precomputed and stored before application execution or generated in real-time during the application's runtime; however, for performance reasons, it is assumed that key generation has been precomputed in a separate preparation phase and stored in storage before the application runs. The secret key, unlike other public keys, must be isolated in secure storage.
- The paths to these keys should be recorded in a configuration file, which also defines a configuration binary (in JSON format or a lighter serialization solution) that contains various settings that may be necessary for the operation of the library. Additionally, a `Configuration` class that parses this file and specifies runtime settings is needed. It includes loaders that load keys into host memory or accelerator memory, and an IO scheduler that decides whether to load all keys at once for performance or load/unload them in real-time considering hardware constraints (e.g., memory size). Public encryption keys can be used within the `belong()` sequence of `Compuon`, and other evaluation keys can be used within the overloaded operator call sequence of `Compuon`.
- Since the public evaluation keys provided by the `KeyManager` are used in RLWE homomorphic encryption schemes like CKKS to correct the distortions relative to the corresponding plaintext results after homomorphic operations (multiplication, rotation, complex conjugation), in some respects, they act as precomputed cheatsheets referenced during the computation process. This means the `KeyManager` must be integrated with the homomorphic encryption library under the actual computing `Backend`.

## **4. Architecture Overview**
**Fhenomenon** is structured into two main components: the **Frontend** and the **Backend**. The **Frontend** offers a user-facing API that abstracts cryptographic complexities, while the **Backend** handles the actual FHE computations. The architecture supports dynamic integration of various FHE libraries through a common interface.

## **4.1 Frontend**
The frontend provides the primary interface for users, allowing them to define and manipulate encrypted entities with minimal cryptographic knowledge. Key components include:
- **Compuon<T>**: A template class(derived from "_Compute + -on_" meaning a computational-capable atomic object) that allows basic C++ types (`int`, `float`, `double`, `complex`) to be encrypted and manipulated as FHE entities. The `Compuon<T>` class maps these types into their corresponding FHE forms based on specified mathematical properties.
- **Session Management**: A Pythonic interface enables users to define code blocks within a session scope. Operations inside the session scope are scheduled and optimized, while those outside the scope are executed immediately. This model promotes a clear and straightforward user experience.
- **Backend Selection**: Users can choose between built-in backends or external libraries during the creation of the backend object. This decision dictates how the library schedules and executes operations.

## **4.2 Backend**
The backend manages the actual FHE operations, either through built-in implementations or by interfacing with external FHE libraries. Two types of backends are supported:
- **BuiltinBackend**: Integrates deeply with the FHE library and offers extended APIs for enhanced parallel processing, including operation fusion and batching. A delegate pattern is used to separate hardware acceleration from core backend logic, allowing for independent development of acceleration kernels.
- **ExternalBackend**: Allows for the dynamic loading of external FHE libraries via shared library paths specified by the user. This backend leverages a common FHE interface to maintain compatibility across different libraries.

## **5. Scheduler Design**
The scheduler is a critical component of **Fhenomenon**, optimizing the execution of FHE operations to enhance performance. It consists of several submodules:
- **Receiver**: Collects API calls from the frontend and prepares them for scheduling.
- **Graph Construction**: Builds a directed graph representing the relationships between ciphertext nodes and operations. This graph serves as the basis for optimization and scheduling strategies.
- **Strategy Module**: Provides a framework for optimization strategies, allowing developers to add new techniques using templates, such as lambda functions. This module aims to maximize data reuse and minimize data I/O, especially when utilizing acceleration hardware.
- **Dispatcher**: Distributes computation tasks to appropriate devices, such as CPUs or GPUs, optimizing resource allocation and performance.

The scheduler is designed to facilitate coarse-grained parallel processing, recognizing computational partitions per ciphertext object and managing them based on their lifecycle. This approach allows **Fhenomenon** to bundle operations efficiently, reducing I/O overhead and maximizing the use of acceleration hardware.

## **6. Key Management**
Key management is a crucial aspect of FHE, where public keys must be accessible, but secret keys need to remain secure. **Fhenomenon** addresses key management with a dedicated module:
- **KeyManager**: Manages the loading and usage of cryptographic keys, ensuring that secret keys are securely isolated while public keys are available for encryption and evaluation operations. Keys are loaded from configuration files, typically defined in JSON format, which specify paths and settings for various keys.
- **IO Scheduler**: Decides whether keys should be preloaded or managed in real-time, balancing performance needs with hardware constraints. This component ensures efficient use of memory and acceleration resources.

## **7. Acceleration and Hardware Detection**
To optimize performance, **Fhenomenon** includes a runtime detector that identifies available hardware accelerators (e.g., GPUs, ASICs). It uses a delegate pattern to bind hardware-specific acceleration kernels independently of the core backend logic, enabling flexible support for various hardware configurations.

## **8. Implementation Details**

## **8.1 Common Interfaces**
At the heart of **Fhenomenon** is a set of common interfaces that define the interaction between the frontend, backend, and scheduler components. These interfaces abstract the specifics of different FHE libraries, providing a unified API for users. Key interfaces include:
- **Frontend Interface**: Defines the basic operations available to users, such as encryption, decryption, and transcryption of `Compuon<T>` objects.
- **Backend Interface**: Outlines the operations required for different FHE backends, including initialization, execution of FHE operations, and interaction with hardware accelerators.
- **Scheduler Interface**: Specifies the methods for constructing and optimizing the operation graph, dispatching tasks, and managing the lifecycle of ciphertext objects.

## **8.2 Backend Implementation**
The **BuiltinBackend** extends the basic backend interface with additional APIs to maximize parallel processing. It uses a delegate pattern to manage hardware-specific acceleration, ensuring that the core backend remains agnostic to the details of the underlying hardware.

The **ExternalBackend** supports dynamic linking of external FHE libraries by loading shared library paths specified by the user. This backend conforms to the common backend interface, ensuring compatibility and ease of integration with various FHE libraries.

## **9. Example Usage**
Below is an example illustrating how a user might interact with **Fhenomenon**:
```cpp
int main() {
    // Initialize the backend as a singleton
    auto backend = BuiltinBackend::Instance();

    // Create a session with the chosen backend
    Session sess(backend);

    // Define operations within the session scope
    sess.run([&]() {
        Compuon<int> a = 10;
        a.belong(rlwe::ckks::configA);
	    auto b = a * 2;
    });

    return 0;
}
```

## **10. Security Considerations**
Security is paramount in FHE applications. **Fhenomenon** ensures the secure handling of cryptographic keys, with secret keys isolated in secure storage and public keys managed through a dedicated KeyManager. The library adheres to best practices for cryptographic software, including secure memory management and avoidance of side-channel vulnerabilities.

## 11. Dependency Diagram
```jsx
+---------------------+
|      Compuon<T>      |
+---------------------+
          |
          v
+---------------------+
|      Session        |
+---------------------+
          |
          v
+-----------------------------+
|       Backend (Interface)   |
+-----------------------------+
        /              \
       /                \
+----------------+   +--------------------+
| BuiltinBackend |   |  ExternalBackend   |
+----------------+   +--------------------+
        |                      |
        | (inherits from)      |
        +----------------------+
                           |
+--------------------------------------------------------+
|          Hardware Abstraction Layer (Delegate)         |
+--------------------------------------------------------+
                           |
                           v
                 +------------------+
                 | Acceleration API |
                 +------------------+
                           |
                           v
+------------------------------+
|     Hardware Detection       |
+------------------------------+
                           |
                           v
+--------------------------------------------------------+
|                      Scheduler                         |
+--------------------------------------------------------+
        |                  |                   |
        |                  |                   |
+------------+     +--------------+     +-------------+
|  Receiver  |<--> |    Graph     |<--> | Dispatcher  |
+------------+     +--------------+     +-------------+
                         |
                         v
              +-------------------+
              |    Strategy       |
              +-------------------+
                         |
                         v
                +----------------+
                |  Optimization  |
                +----------------+
                         |
                         v
+--------------------------------------------------------+
|                   KeyManager                           |
+--------------------------------------------------------+
                         |
                         v
+----------------+     +----------------+
|    Key I/O     |<--> |    Config      |
+----------------+     +----------------+

```
