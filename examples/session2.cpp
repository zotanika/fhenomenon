#include "Fhenomenon.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>


#define AaBbCc "sword"
#define XxYyZz "potion"
using namespace fhenomenon;

std::string encryptDecrypt(std::string to_encrypt) {
    char key = 'K';
    std::string output = to_encrypt;
    
    for (int i = 0; i < to_encrypt.size(); i++)
        output[i] = to_encrypt[i] ^ key;
    
    return output;
}
template<typename T>
class Helix{
private:
    Compuon<T> entity_;
    std::shared_ptr<Profile> profile_;

public:
    inline Helix(T value = T()): entity_(Compuon<T>(value)){
        profile_ = Profile::getProfile();
        entity_.belong(profile_);
    }

    inline Helix(const Compuon<T>& entity): entity_(entity){
        profile_ = Profile::getProfile();
    }

    inline ~Helix() = default;

    inline Helix& operator=(const Helix& other) {
        entity_ = other.entity_;
        return *this;     
    }
    inline Helix& operator=(const T& num) {
        auto scalar = Helix<T>(num);
        entity_ = scalar.entity_;
        return *this;
    }
    inline Helix operator+(const Helix& other) const {
        Helix<T> result = entity_ + other.entity_;
        return result;
    }

    inline Helix operator+(const T& num) const {
        return *this + Helix<T>(num);  
    }

    T getValue() const{
        return this->entity_.decrypt();
    }
};

class HEInventoryManager {
private:
    std::map<std::string, Helix<int>> inventory_;
public:
    void addItem(const std::string& item, int count) {
        if (inventory_.find(item) == inventory_.end())
            inventory_[item] = 0;
        inventory_[item] = inventory_[item] + count;
    }
    
    void showInventory() {
        for (auto& item : inventory_) {
            std::cout << encryptDecrypt(item.first) << ": " << item.second.getValue() << std::endl;
        }
    }
};

int main() {
    std::shared_ptr<Parameter> param_;
    std::shared_ptr<Profile> profile_ = Profile::createProfile(param_);
    HEInventoryManager manager;
    char choice;
    int dummy_var = 0;
    const int dummy_mask = 0xCC;

    while (true) {
        std::cout << "1. Add Sword\n";
        std::cout << "2. Add Potion\n";
        std::cout << "3. Show Inventory\n";
        std::cout << "4. Exit\n";
        std::cout << "Enter choice: ";
        std::cin >> choice;

        switch(choice) {
            case '1':
                manager.addItem(AaBbCc, 1);
                dummy_var++;
                break;
            case '2':
                manager.addItem(XxYyZz, 1);
                dummy_var--;
                break;
            case '3':
                manager.showInventory();
                dummy_var ^= dummy_mask;
                break;
            case '4':
                return 0;
            default:
                std::cout << "Invalid choice\n";
                break;
        }
    }

    return 0;
}