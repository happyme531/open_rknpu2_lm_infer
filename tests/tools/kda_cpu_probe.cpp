#include "kda_candidate.h"
#include "kda_reference.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using Kernel=void(*)(const rknpu2_kda_params*);
std::vector<float> read(const std::filesystem::path& file) {
    std::ifstream f(file,std::ios::binary|std::ios::ate);
    if(!f)throw std::runtime_error("open "+file.string());
    const auto size=f.tellg();if(size%4)throw std::runtime_error("invalid float file");
    std::vector<float> out(size_t(size)/4);f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),size);
    if(!f)throw std::runtime_error("read "+file.string());return out;
}
bool equal(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*4)==0;
}
int main(int argc,char** argv)try {
    if(argc!=2)throw std::runtime_error("usage: kda_cpu_probe CAPTURE_DIRECTORY");
    std::cout<<std::setprecision(10)<<"{\"cases\":[";bool comma=false;
    const char* names[]={"baseline","row","block4","tile8","production"};
    Kernel kernels[]={kda_reference,kda_probe::run<1,1>,kda_probe::run<4,1>,kda_probe::run<4,8>,rknpu2_kda_run};
    constexpr int variants=5;
    for(const auto& fixture:{"m128_zero","m128_nonzero","m1_nonzero"}) {
        const auto dir=std::filesystem::path(argv[1])/fixture;
        const auto initial=read(dir/"state.f32"),q=read(dir/"q.f32"),k=read(dir/"k.f32"),
            value=read(dir/"v.f32"),decay=read(dir/"decay.f32"),beta=read(dir/"beta.f32");
        const size_t rows=q.size()/2048;
        for(size_t threads:{1u,4u}) {
            auto expected_state=initial;std::vector<float> expected_output(q.size());
            rknpu2_kda_params p{rows,16,128,threads,expected_state.data(),q.data(),k.data(),value.data(),decay.data(),beta.data(),expected_output.data()};
            kda_reference(&p);
            for(auto kernel:kernels) {
                auto state=initial;std::vector<float> output(q.size());p.state=state.data();p.output=output.data();kernel(&p);
                if(!equal(state,expected_state)||!equal(output,expected_output))throw std::runtime_error(std::string("nonidentical ")+fixture);
            }
            const int loops=rows==1?512:8;
            std::vector<double> times[variants];
            std::vector<float> states[variants],outputs[variants];
            for(int v=0;v<variants;++v){states[v]=initial;outputs[v].resize(q.size());p.state=states[v].data();p.output=outputs[v].data();kernels[v](&p);}
            for(int round=0;round<8;++round)for(int j=0;j<variants;++j) {
                const int v=(round%2==0)?j:variants-1-j;
                states[v]=initial;p.state=states[v].data();p.output=outputs[v].data();
                const auto start=std::chrono::steady_clock::now();
                for(int i=0;i<loops;++i)kernels[v](&p);
                times[v].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/loops);
            }
            for(int v=1;v<variants;++v)if(!equal(states[0],states[v])||!equal(outputs[0],outputs[v]))throw std::runtime_error("repeated state/output mismatch");
            if(comma)std::cout<<',';comma=true;
            std::cout<<"{\"fixture\":\""<<fixture<<"\",\"rows\":"<<rows<<",\"threads\":"<<threads<<",\"loops\":"<<loops<<",\"exact\":true,\"samples_us\":{";
            for(int v=0;v<variants;++v){if(v)std::cout<<',';std::cout<<'"'<<names[v]<<"\":[";for(size_t i=0;i<times[v].size();++i){if(i)std::cout<<',';std::cout<<times[v][i];}std::cout<<']';}
            std::cout<<"}}"<<std::flush;
        }
    }
    std::cout<<"]}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
