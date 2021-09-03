#include <string>
#include <chrono>
#include <iostream>

using namespace std;

class ScopeTimer
{
public:
    ScopeTimer(string msg)
    {
        msg_ = msg;
        startPoint_ = chrono::high_resolution_clock::now();
    }

    ~ScopeTimer()
    {
        auto duration = chrono::duration_cast<chrono::minutes>(chrono::high_resolution_clock::now() - startPoint_).count();

        cout << msg_ << " " << duration << endl;
    }

private:
    string msg_;
    chrono::time_point<std::chrono::high_resolution_clock> startPoint_;
};