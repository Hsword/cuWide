#include <cstdio>
#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
using namespace std;

//#include <cuda_runtime.h>
//#include <cooperative_groups.h>
//using namespace cooperative_groups;

const int NUM_THREADS=1024;
const int CLK_TCK=CLOCKS_PER_SEC;
const int mini_batch_size = 32;
const int sm_w_size = 250;
const int num_blocks = 56;
const int early_stop = 10;
#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { printf ("\nCUDA Error: %s (err_num=%d) \n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);} }
#define NUM_REPLICA 4
#define NUM_B_REPLICA 32
#define WARP 0
int* Replace_Vec;
int* Inverse_Replace_Vec;

bool cmp_pair(const pair<int,long long> a, const pair<int,long long> b)
{
  return a.second>b.second;
}

struct Item{
  int sample_id;
  float val;
};

struct Matrix{
  int *sample_id;
  int *feature_id;
  float *val;
  int num;
  int *label;
  float *err;
  int batch_size;
};

void MatrixMalloc(Matrix *m, int batch_size, int num)
{
  CUDA_CALL(cudaMalloc((int**)&m->sample_id, num * sizeof(int)));
  CUDA_CALL(cudaMalloc((int**)&m->feature_id, num * sizeof(int)));
  CUDA_CALL(cudaMalloc((float**)&m->val, num * sizeof(float)));
  CUDA_CALL(cudaMalloc((int**)&m->label, batch_size * sizeof(int)));
  CUDA_CALL(cudaMalloc((float**)&m->err, batch_size * sizeof(float)));
  m->num=num;
  m->batch_size=batch_size;
}

void MatrixFree(Matrix *m)
{
  CUDA_CALL(cudaFree(m->sample_id));
  CUDA_CALL(cudaFree(m->feature_id));
  CUDA_CALL(cudaFree(m->val));
  CUDA_CALL(cudaFree(m->label));
  CUDA_CALL(cudaFree(m->err));
}

struct MatrixHost
{
    int *sample_id;
    int *feature_id;
    float *val;
    int num;
    int *label;
    int batch_size;
};

void MatrixHostMalloc(MatrixHost *m, int batch_size, int num)
{
  CUDA_CALL(cudaMallocHost((int**)&m->sample_id, num * sizeof(int)));
  CUDA_CALL(cudaMallocHost((int**)&m->feature_id, num * sizeof(int)));
  CUDA_CALL(cudaMallocHost((float**)&m->val, num * sizeof(float)));
  CUDA_CALL(cudaMallocHost((int**)&m->label, batch_size * sizeof(int)));
  m->num=num;
  m->batch_size=batch_size;
}

void MatrixHostFree(MatrixHost *m)
{
  CUDA_CALL(cudaFreeHost(m->sample_id));
  CUDA_CALL(cudaFreeHost(m->feature_id));
  CUDA_CALL(cudaFreeHost(m->val));
  CUDA_CALL(cudaFreeHost(m->label));
}

void MatrixCopyHostToDevice(Matrix *m, MatrixHost *m_host)
{
  CUDA_CALL(cudaMemcpy(m->sample_id, m_host->sample_id, m->num*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(m->feature_id, m_host->feature_id, m->num*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(m->val, m_host->val, m->num*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(m->label, m_host->label, m->batch_size*sizeof(int), cudaMemcpyHostToDevice));
  m->batch_size=m_host->batch_size;
}

__global__ void calc_grad(int* sample_id, int* feature_id, float *val, int* label, float* w, int batch_size, float learning_rate, float* err, int* start_index_mini_batch)
{
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float sm_product[mini_batch_size*NUM_REPLICA];
  __shared__ float sm_w[sm_w_size*NUM_B_REPLICA];
  int mini_batch_num = batch_size/mini_batch_size+((batch_size%mini_batch_size)?1:0);
  for(int k=block_id; k<=mini_batch_num; k+=num_blocks)
  { 
    // Aggregate
    __syncthreads();
    for(int thread_index = thread_id; thread_index<(mini_batch_size*NUM_REPLICA); thread_index+=blockDim.x)
      sm_product[thread_index]=0;
    __syncthreads();
    int upper_bound = start_index_mini_batch[2*k+2];
    int upper_bound_w = start_index_mini_batch[2*k+1];
    for(int i=start_index_mini_batch[k*2]+thread_id;i<upper_bound;i+=blockDim.x)
    {
      int fid = feature_id[i];
      int sid = sample_id[i]-k*mini_batch_size;
      float v = val[i];
      atomicAdd(&sm_product[sid*NUM_REPLICA+thread_id%NUM_REPLICA], w[fid]*v);
    }
    __syncthreads();
    //reduce for 32
    if(NUM_REPLICA!=1)
    {
    if(WARP)
    {
      for(int i=thread_id;i<NUM_REPLICA*mini_batch_size;i+=blockDim.x)
      {
        float sum=sm_product[i];
        sum+=__shfl_down(sum, 16);
        sum+=__shfl_down(sum, 8);
        sum+=__shfl_down(sum, 4);
        sum+=__shfl_down(sum, 2);
        sum+=__shfl_down(sum, 1);
        sm_product[i]=sum;
      }
    }
    else
    { 
    if(thread_id<mini_batch_size)
    {
      int id = thread_id%NUM_REPLICA+thread_id*NUM_REPLICA;
      float tmp_product=sm_product[id];
      for(int i=(thread_id%NUM_REPLICA+1)%NUM_REPLICA+thread_id*NUM_REPLICA;i!=id;i=(i+1)%NUM_REPLICA+thread_id*NUM_REPLICA)
      {
        tmp_product+=sm_product[i];
      }
      sm_product[id]=tmp_product;
      for(int i=(thread_id%NUM_REPLICA+1)%NUM_REPLICA+thread_id*NUM_REPLICA;i!=id;i=(i-thread_id*NUM_REPLICA+1)%NUM_REPLICA+thread_id*NUM_REPLICA)
      {
        sm_product[i]=tmp_product;
      } 
    }
    }
    }
    __syncthreads();
    // Loss & Apply
    for(int thread_index = thread_id; thread_index<(sm_w_size*NUM_B_REPLICA); thread_index+=blockDim.x)
      sm_w[thread_index]=0;
    __syncthreads();
    for(int i=start_index_mini_batch[2*k]+thread_id;i<upper_bound_w;i+=blockDim.x)
    {
      int fid = feature_id[i];
      int sid = sample_id[i];
      float v = val[i];
      float product = sm_product[(sid-k*mini_batch_size)*NUM_REPLICA+thread_id%NUM_REPLICA];
      float error = label[sid]-1.0/(1+exp(-product));
      err[sid] = -(label[sid]-(product>0))*product+log(1+exp(product-2*product*(product>0)));
      sm_w[fid*NUM_B_REPLICA+thread_id%NUM_B_REPLICA]+=learning_rate*error*v/mini_batch_size;
    }
    __syncthreads();
    if(NUM_B_REPLICA!=1)
    {
    for(int i=NUM_B_REPLICA/2;i>0;i>>=1)
    {
      for(int tid=thread_id; tid/i<sm_w_size; tid+=blockDim.x)
      {
        int index = tid/i*NUM_B_REPLICA+tid%i;
        sm_w[index]+=sm_w[index+i]; 
      }
      __syncthreads();
    }
    }
    __syncthreads();
    if(thread_id<sm_w_size)
      atomicAdd(&w[thread_id], sm_w[thread_id*NUM_B_REPLICA]);
    __syncthreads();
    for(int i=upper_bound_w+thread_id;i<upper_bound;i+=blockDim.x)
    {
      int fid = feature_id[i];
      int sid = sample_id[i];
      float v = val[i];
      float product = sm_product[(sid-k*mini_batch_size)*NUM_REPLICA+thread_id%NUM_REPLICA];
      float error = label[sid]-1.0/(1+exp(-product));
      err[sid] = -(label[sid]-(product>0))*product+log(1+exp(product-2*product*(product>0)));
      atomicAdd(&w[fid], learning_rate*error*v/mini_batch_size);
    }
  }
}
void batch_lr(Matrix *batch, float* w, float learning_rate, int* start_index_mini_batch)
{
    calc_grad<<<num_blocks,NUM_THREADS>>>(batch->sample_id,batch->feature_id,batch->val,batch->label,w,batch->batch_size,learning_rate,batch->err,start_index_mini_batch);
    CUDA_CALL(cudaDeviceSynchronize()); 
}

void convert_2_coo(MatrixHost *mat_host, int mini_batch_num, vector <vector <pair <int, float> > > &Data, vector <int> &Label, int *start_index_mini_batch)
{
  int *start_index_mini_batch_host = (int*)malloc((2*mini_batch_num+1)*sizeof(int));
  int mini_batch_index=0;
  int tmp_num=0;
  mat_host->batch_size=0;
  for(int i=0;i<Data.size();i++)
  {
    if(i%mini_batch_size==0)
    {
      start_index_mini_batch_host[mini_batch_index]=tmp_num;
      mini_batch_index+=2;
    }
    for(int j=0;j<Data[i].size();j++)
    {
        mat_host->sample_id[tmp_num]=i;
        mat_host->feature_id[tmp_num]=Inverse_Replace_Vec[Data[i][j].first];
        mat_host->val[tmp_num]=Data[i][j].second;
        mat_host->label[i]=Label[i];
        tmp_num++;
    }
    mat_host->batch_size++;
  }
  start_index_mini_batch_host[mini_batch_index]=tmp_num;
  //sort by feature id
  for(int i=0; i<mini_batch_index-1; i+=2)
  {
    int st = start_index_mini_batch_host[i];
    int ed = start_index_mini_batch_host[i+2];
    map <int, vector <Item> > MiniBatch;
    map <int, vector <Item> >::iterator iter;
    MiniBatch.clear();
    for(int j=st; j<ed; j++)
    {
      iter = MiniBatch.find(mat_host->feature_id[j]);
      Item item;
      item.sample_id = mat_host->sample_id[j];
      item.val = mat_host->val[j];
      if(iter==MiniBatch.end())
      {
        vector<Item> tmp;
        tmp.push_back(item);
        MiniBatch[mat_host->feature_id[j]]=tmp;
      }
      else
      {
        (iter->second).push_back(item);
      }
    }
    int tmp_cnt=st;
    bool set_index = false;
    for(iter=MiniBatch.begin();iter!=MiniBatch.end();iter++)
    {
      if((!set_index)&&(iter->first>=sm_w_size))
      {
        start_index_mini_batch_host[i+1]=tmp_cnt;
        set_index=true;
      }
      for(vector<Item>::iterator viter=iter->second.begin();viter!=iter->second.end();viter++)
      {
        mat_host->feature_id[tmp_cnt]=iter->first;
        mat_host->sample_id[tmp_cnt]=(*viter).sample_id;
        mat_host->val[tmp_cnt++]=(*viter).val;
      }
    }
    if(!set_index)
    {
      start_index_mini_batch_host[i+1]=ed;
    }
  }
  CUDA_CALL(cudaMemcpy(start_index_mini_batch,start_index_mini_batch_host,(2*mini_batch_num+1)*sizeof(int),cudaMemcpyHostToDevice));
  free(start_index_mini_batch_host);
}


__global__ void spmv_kernel(int *sample_id,int *feature_id,float *val,int *label,float *w,float *product,int num, float *err, int batch_size)
{
    for(int i= blockIdx.x * blockDim.x + threadIdx.x;i<num;i+=gridDim.x*blockDim.x)
    {
      int fid = feature_id[i];
      int sid = sample_id[i];
      float v = val[i];
      atomicAdd(&product[sid], w[fid]*v);
    }
}


__global__ void backward_kernel(int *sample_id,int *feature_id,float *val,int *label,float *w,float *product,int num, float *err, int batch_size)
{
    for(int i=blockIdx.x * blockDim.x + threadIdx.x;i<batch_size;i+=gridDim.x*blockDim.x)
    {
      float prod = product[i];
      err[i]= -(label[i]-(prod>0))*prod+log(1+exp(prod-2*prod*(prod>0)));
    }
}


void test_lr(Matrix *batch, float* w, int batch_size)
{
    float *product;
    CUDA_CALL(cudaMalloc((float**)&product, batch_size * sizeof(float)));
    CUDA_CALL(cudaMemset(product, 0, batch_size * sizeof(float)));
    void **args=(void**)malloc(9*sizeof(void*));
    args[0]=&batch->sample_id;
    args[1]=&batch->feature_id;
    args[2]=&batch->val;
    args[3]=&batch->label;
    args[4]=&w;
    args[5]=&product;
    args[6]=&batch->num;
    args[7]=&batch->err;
    args[8]=&batch_size;
    spmv_kernel<<<num_blocks,NUM_THREADS>>>(batch->sample_id,batch->feature_id,batch->val,batch->label,w,product, batch->num,batch->err,batch_size);
    backward_kernel<<<num_blocks,NUM_THREADS>>>(batch->sample_id,batch->feature_id,batch->val,batch->label,w,product, batch->num,batch->err,batch_size);
    CUDA_CALL(cudaFree(product));
}

float avg_val(vector <float> cost_val, int n)
{
   float sum=0;
   int l = cost_val.size();
   for(int i=0; i<n; i++)
     sum+=cost_val[l-1-i];
   return sum/n;
}


void sgd(vector <vector <pair <int, float> > > &Data, vector <int> &Label, int max_iterations, float learning_rate, int data_size, int feature_num, int num, vector <vector <pair <int, float> > > &TestData, vector <int> &TestLabel, int test_size, int test_num)
{
    int mini_batch_num = data_size/mini_batch_size+((data_size%mini_batch_size)?1:0);
    float *w;
    CUDA_CALL(cudaMalloc((float**)&w, feature_num * sizeof(float)));
    CUDA_CALL(cudaMemset(w, 0, feature_num * sizeof(float)));
    float *err = (float*)malloc(data_size*sizeof(float));
    float *test_err = (float*)malloc(test_size*sizeof(float));
    MatrixHost mat_host;
    MatrixHostMalloc(&mat_host, data_size, num);
    int *start_index_mini_batch;
    CUDA_CALL(cudaMalloc((int**)&start_index_mini_batch, (2*mini_batch_num+1)*sizeof(int)));
    Matrix mat;
    MatrixMalloc(&mat, data_size, num);
    clock_t start,end;
    start = clock();
    printf("converting data to COO format\n");
    convert_2_coo(&mat_host, mini_batch_num, Data, Label, start_index_mini_batch);
    end = clock(); 
    printf("convert data to COO format time cost: %.2f\n",double(end-start)/CLK_TCK);
    start = clock();
    printf("move data to GPU memory\n");
    MatrixCopyHostToDevice(&mat, &mat_host);
    end = clock();
    printf("move data to GPU memory time cost: %.2f\n",double(end-start)/CLK_TCK);

    //Test Data 
    Matrix test_mat;
    MatrixMalloc(&test_mat, test_size, test_num);
    MatrixHostMalloc(&mat_host, test_size, test_num);
    int tmp_coo_index = 0;
    for(int i = 0; i < test_size; i++)
    {
      mat_host.label[i] = TestLabel[i];
      for(int j = 0; j < TestData[i].size(); j++)
      {
        mat_host.sample_id[tmp_coo_index] = i;
        mat_host.feature_id[tmp_coo_index] = Inverse_Replace_Vec[TestData[i][j].first];
        mat_host.val[tmp_coo_index] = TestData[i][j].second;
        tmp_coo_index++;
      }
    }
    MatrixCopyHostToDevice(&test_mat, &mat_host);
    MatrixHostFree(&mat_host);

    vector<float> cost_val;
    timeval StartingTime, PausingTime, ElapsedTime;
    int train_time = 0;
    for(int i=0;i<max_iterations;i++)
    {
        start = clock();
        gettimeofday(&StartingTime, NULL);
        batch_lr(&mat, w, learning_rate, start_index_mini_batch);
        gettimeofday(&PausingTime, NULL);
        timersub(&PausingTime, &StartingTime, &ElapsedTime);
        train_time += ElapsedTime.tv_sec * 1000.0 * 1000.0 + ElapsedTime.tv_usec;
        CUDA_CALL(cudaMemcpy(err, mat.err, sizeof(float)*mat.batch_size, cudaMemcpyDeviceToHost));
        float sum=0;
        for(int j=0;j<mat.batch_size;j++)
            sum+=err[j];
        end = clock();
        printf("iter: %d \ttraining loss: %.6f\ttime cost: %.2fs\n",i+1,sum/mat.batch_size,double(end-start)/CLK_TCK);
        test_lr(&test_mat, w, test_size);
        CUDA_CALL(cudaMemcpy(test_err, test_mat.err, sizeof(float)*test_size, cudaMemcpyDeviceToHost));
        sum=0;
        for(int j=0;j<test_size;j++)
        {
          sum+=test_err[j];
        }
        float sum_err = sum/test_size;
        printf("Time: %d\tTest Loss: %f\n",train_time, sum_err);
        cost_val.push_back(sum_err);
        if(i>early_stop && sum_err>avg_val(cost_val, early_stop))
          break;
    }
    MatrixFree(&mat);
    CUDA_CALL(cudaFree(w));
    CUDA_CALL(cudaFree(start_index_mini_batch));
    free(err);
}

inline char get_next(char* &buf, char *end)
{
    if(buf<end) return *buf++;
    else return EOF;
}

inline bool get_float(char* end, char* &buf, float &num)
{
    char in;double Dec=0.1;
    bool IsN=false,IsD=false;
    in=get_next(buf, end);
    if(in==EOF) return false;
    while(in!='-'&&in!='.'&&(in<'0'||in>'9'))
    {
      in=get_next(buf, end);
      if(in==EOF) return false;
    }
    if(in=='-'){IsN=true;num=0;}
    else if(in=='.'){IsD=true;num=0;}
    else num=in-'0';
    if(!IsD){
        while(in=get_next(buf, end),in>='0'&&in<='9'){
            num*=10;num+=in-'0';}
    }
    if(in!='.'){
        if(IsN) num=-num;
        buf--;
        return true;
    }else{
        while(in=get_next(buf, end),in>='0'&&in<='9'){
            num+=Dec*(in-'0');Dec*=0.1;
        }
    }
    if(IsN) num=-num;
    buf--;
    return true;
}

inline bool get_int(char* end, char* &buf, int &num)
{
    char in;bool IsN=false;
    in=get_next(buf, end);
    if(in==EOF) return false;
    while(in!='-'&&(in<'0'||in>'9'))
    {
        in=get_next(buf, end);
        if(in==EOF) return false;
    }
    if(in=='-'){ IsN=true;num=0;}
    else num=in-'0';
    while(in=get_next(buf, end),in>='0'&&in<='9'){
        num*=10,num+=in-'0';
    }
    if(IsN) num=-num;
    buf--;
    return true;
}

int main( void ) {
    map <int, long long> Feature_Dict;
    map <int, long long>::iterator iter;
    vector <pair <int, long long> > Feature_Dict_Vec;
    vector <vector <pair <int, float> > > Data;
    vector <vector <pair <int, float> > > TestData;
    vector <int> Label;
    vector <int> TestLabel;
    int data_size = 0;
    int test_size = 0;
    int total_size = 0;
    int feature_num = 0;
    int fd = open("./criteo_sample.dat",O_RDONLY);
    long long file_len = lseek(fd,0,SEEK_END);
    char *file_buf = (char *) mmap(NULL,file_len,PROT_READ,MAP_PRIVATE,fd,0);
    char *file_end= file_buf+file_len;
    int num = 0;
    int test_num = 0;
    int label;
    float feature;
    int feature_id;
    clock_t start,end;
    clock_t st,en;
    st = clock();
    printf("loading data\n");
    start = clock();
    bool train_or_test = true;
    while(get_int(file_end, file_buf, label))
    {
        if(total_size % 10 == 0)
          train_or_test = false;
        else
          train_or_test = true;
        if(train_or_test)
          Label.push_back(label);
        else
          TestLabel.push_back(label);
        vector<pair<int,float> > v;
        do{
            get_int(file_end, file_buf, feature_id);
            get_float(file_end, file_buf, feature);
            feature_num = max(feature_num, feature_id);
            v.push_back(make_pair<int,float>(feature_id,feature));
            iter = Feature_Dict.find(feature_id);
            if(iter != Feature_Dict.end())
              iter->second=iter->second+1;
            else
              Feature_Dict.insert(pair<int,long long>(feature_id,1)); 
            if(train_or_test)
              num++;
            else
              test_num++;
        }while((*file_buf)!='\n'&&file_buf<file_end); 
        if(train_or_test)
        {
          Data.push_back(v);
          data_size++;
        }
        else
        {
          TestData.push_back(v);
          test_size++;
        }
        total_size++;
    }
    feature_num++;
    for(int i = 0; i < feature_num; i++)
    {
      iter = Feature_Dict.find(i);
      if(iter == Feature_Dict.end())
        Feature_Dict.insert(pair<int,long long>(i, 0)); 
    }
    end = clock();
    cout<<"feature_num="<<feature_num<<" coo_item_num="<<num<<" data_size="<<data_size<<endl;
    printf("load data time cost: %.2fs\n",double(end-start)/CLK_TCK);
    start = clock();
    for(iter = Feature_Dict.begin(); iter != Feature_Dict.end(); iter++)
    {
      Feature_Dict_Vec.push_back(pair<int,long long>(iter->first,iter->second));
    }
    sort(Feature_Dict_Vec.begin(), Feature_Dict_Vec.end(), cmp_pair);
    Replace_Vec = new int[feature_num];
    Inverse_Replace_Vec = new int[feature_num]; 
    for(int i=0; i < feature_num; i++)
    {
      int index =  Feature_Dict_Vec[i].first;
      Replace_Vec[i] = index;
      Inverse_Replace_Vec[index] = i;
    }
    end = clock();
    cout<<"build index time cost: "<<(double)(end-start)/CLK_TCK<<endl;
    sgd(Data, Label, 100, 0.5, data_size, feature_num, num, TestData, TestLabel, test_size, test_num);
    en = clock();
    printf("Time cost: %.2fs\n",double(en-st)/CLK_TCK);
    return 0;
}
