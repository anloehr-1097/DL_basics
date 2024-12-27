#include <string>
#include <tuple>
#include <iostream>

template<typename... Args>
std::string tuple_to_string(std::tuple<Args...>& t) {
    return std::apply([](const auto&... args) {
        return ((std::to_string(args) + ' ') + ...);
    }, t);
}
