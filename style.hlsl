#pragma kernel malloc_particle
#pragma kernel clear_hash_grid
#pragma kernel compute_hash_grid
#pragma kernel compute_neighbor_list
#pragma kernel compute_density_pressure
#pragma kernel compute_force
#pragma kernel integrate

struct particle
{
    float3 position;
    float4 color;
};

RWStructuredBuffer<particle> particles;
RWStructuredBuffer<float> density,
pressure,
bound;
RWStructuredBuffer<float3> force,
velocity;
RWStructuredBuffer<int> neighbor_list, /* neighbors of a particle at particle_index * max_particles_per_grid * n */
neighbor_tracker; /* number of neighors does each particle have */
RWStructuredBuffer<uint> hash_grid, /* aligned at particle_index * max_particles_per_grid * n' + hash_grid_tracker[particle_index] */
hash_grid_tracker; /* number of particles at each grid */

float grid_size,
radius,
radius2,
radius3,
radius4,
radius5,
mass,
mass2,
gas_constant,
rest_density,
viscosity_coefficient,
damping,
dt,
g,
epsilon,
pi;
uint n,
n_particle,
dimension,
max_particles_per_grid;
float4 time;

int3 get_cell(float3 position)
{
    return int3(position.x / grid_size, position.y / grid_size, position.z / grid_size);
}

int hash(int3 cell)
{
    return cell.x + dimension * (cell.y + dimension * cell.z);
}

[numthreads(1, 1, 1)]
void malloc_particle(uint3 id : SV_DispatchThreadID)
{
    uint particle_per_dimension = pow(n_particle, 1.0 / 3.0),
    i = 0;
    while(i < n_particle)
    {
        for(uint x = 0; x < particle_per_dimension; ++x)
            for(uint y = 0; y < particle_per_dimension; ++y)
                for(uint z = 0; z < particle_per_dimension; ++z)
                {
                    float3 pos = float3(dimension - 1, dimension - 1, dimension - 1);
                    particles[i].position = pos;
                    particles[i].color = float4(0.25, 0.5, 1, 0.5);
                    if(++i == n_particle) return;
                }
    }
}

[numthreads(100, 1, 1)]
void clear_hash_grid(uint3 id : SV_DispatchThreadID)
{
    hash_grid_tracker[id.x] = 0;
}

[numthreads(100, 1, 1)]
void compute_hash_grid(uint3 id : SV_DispatchThreadID)
{
    uint original_value = 0;
    const int hash_grid_position = hash(get_cell(particles[id.x].position));
    InterlockedAdd(hash_grid_tracker[hash_grid_position], 1, original_value); /* 'original_value' is an output variable that will be set to the original value of dest */
    if(original_value >= max_particles_per_grid)
    {
        particles[id.x].color = float4(0.25, 0.5, 1, 0.5);
        return;
    }
    hash_grid[hash_grid_position * max_particles_per_grid + original_value] = id.x;
}

void set_neighbor_key(int3 origin_index, float3 position, out int neighbor_key[8])
{
    int3 neighbor_index[8];
    for(uint i = 0; i < n; ++i)
        neighbor_index[i] = origin_index;
    if((origin_index.x + 0.5f) * grid_size <= position.x)
    {
        ++neighbor_index[4].x;
        ++neighbor_index[5].x;
        ++neighbor_index[6].x;
        ++neighbor_index[7].x;
    }
    else
    {
        --neighbor_index[4].x;
        --neighbor_index[5].x;
        --neighbor_index[6].x;
        --neighbor_index[7].x;
    }
    if((origin_index.y + 0.5f) * grid_size <= position.y)
    {
        ++neighbor_index[2].y;
        ++neighbor_index[3].y;
        ++neighbor_index[6].y;
        ++neighbor_index[7].y;
    }
    else
    {
        --neighbor_index[2].y;
        --neighbor_index[3].y;
        --neighbor_index[6].y;
        --neighbor_index[7].y;
    }
    if((origin_index.z + 0.5f) * grid_size <= position.z)
    {
        ++neighbor_index[1].z;
        ++neighbor_index[3].z;
        ++neighbor_index[5].z;
        ++neighbor_index[7].z;
    }
    else
    {
        --neighbor_index[1].z;
        --neighbor_index[3].z;
        --neighbor_index[5].z;
        --neighbor_index[7].z;
    }
    for(uint a = 0; a < n; ++a)
    {
        uint3 neighbor = neighbor_index[a];
        if(neighbor.x < 0 || neighbor.x >= dimension || neighbor.y < 0 || neighbor.y >= dimension || neighbor.z < 0 || neighbor.z >= dimension)
            neighbor_key[a] = -1;
        else
            neighbor_key[a] = hash(neighbor_index[a]);
    }
}

[numthreads(100, 1, 1)]
void compute_neighbor_list(uint3 id : SV_DispatchThreadID)
{
    neighbor_tracker[id.x] = 0;
    const int3 g = get_cell(particles[id.x].position);
    int grids[8];
    set_neighbor_key(g, particles[id.x].position, grids);
    for(uint i = 0; i < n; ++i)
    {
        if(grids[i] == -1) continue;
        const uint particle_in_grid = min(hash_grid_tracker[grids[i]], max_particles_per_grid);
        for(uint j = 0; j < particle_in_grid; ++j)
        {
            const uint potential_neighbor = hash_grid[grids[i] * max_particles_per_grid + j];
            if(potential_neighbor == id.x) continue;
            const float3 v = particles[potential_neighbor].position - particles[id.x].position;
            if(dot(v, v) < radius2)
                neighbor_list[id.x * max_particles_per_grid * n + neighbor_tracker[id.x]++] = potential_neighbor;
        }
    }
}

float std_kernel(float distance_square)
{
    float x = 1.0 - distance_square / radius2;
    return 315.0 / (64.0 * pi * radius3) * x * x * x;
}

float spiky_kernel_first_derivative(float distance)
{
    float x = 1.0 - distance / radius;
    return -45.0 / (pi * radius4) * x * x;
}

float spiky_kernel_second_derivative(float distance)
{
    float x = 1.0 - distance / radius;
    return 90.0 / (pi * radius5) * x;
}

float3 spiky_kernel_gradient(float distance, float3 direction_from_center)
{
    return spiky_kernel_first_derivative(distance) * direction_from_center;
}

[numthreads(100, 1, 1)]
void compute_density_pressure(uint3 id : SV_DispatchThreadID)
{
    float3 origin = particles[id.x].position;
    float sum = 0;
    for(int i = 0; i < neighbor_tracker[id.x]; ++i)
    {
        int neighbor_index = neighbor_list[id.x * max_particles_per_grid * n + i];
        float3 diff = origin - particles[neighbor_index].position;
        float distance_square = dot(diff, diff);
        sum += std_kernel(distance_square);
    }
    density[id.x] = sum * mass + 0.000001f;
    pressure[id.x] = gas_constant * (density[id.x] - rest_density);
}

[numthreads(100, 1, 1)]
void compute_force(uint3 id : SV_DispatchThreadID)
{
    force[id.x] = float3(0, 0, 0);
    float particle_density2 = density[id.x] * density[id.x];
    for(int i = 0; i < neighbor_tracker[id.x]; ++i)
    {
        int neighbor_index = neighbor_list[id.x * max_particles_per_grid * n + i];
        float distance = length(particles[id.x].position - particles[neighbor_index].position);
        if(distance > 0)
        {
            float3 direction = (particles[id.x].position - particles[neighbor_index].position) / distance;
            force[id.x] -= mass2 * (pressure[id.x] / particle_density2 + pressure[neighbor_index] / (density[neighbor_index] * density[neighbor_index])) * spiky_kernel_gradient(distance, direction) + 1000 * noise(float3(time.x, time.y, time.z)); /* compute pressure gradient force */
            force[id.x] += viscosity_coefficient * mass2 * (velocity[neighbor_index] - velocity[id.x]) / density[neighbor_index] * spiky_kernel_second_derivative(distance) + 1000 * noise(float3(time.x, time.y, time.z));
        }
    }
    force[id.x] += g;
}

[numthreads(100, 1, 1)]
void integrate(uint3 id : SV_DispatchThreadID)
{
    particle particle = particles[id.x];
    velocity[id.x] += dt * force[id.x] / mass;
    particle.position += dt * velocity[id.x];
    particles[id.x] = particle;
    particle = particles[id.x];
    float3 v = velocity[id.x];
    if(particles[id.x].position.x < bound[0] + epsilon)
    {
        v.x *= damping;
        particle.position.x = bound[0] + epsilon + 0.3 * noise(particle.position);
    }
    else if(particles[id.x].position.x > bound[1] - epsilon) 
    {
        v.x *= damping;
        particle.position.x = bound[1] - epsilon - 0.3 * noise(particle.position);
    }
    if(particles[id.x].position.y < bound[2] + epsilon)
    {
        v.y *= damping;
        particle.position.y = bound[2] + epsilon + 0.3 * noise(particle.position);
    }
    else if(particles[id.x].position.y > bound[3] - epsilon) 
    {
        v.y *= damping;
        particle.position.y = bound[3] - epsilon - 0.3 * noise(particle.position);
    }
    if(particles[id.x].position.z < bound[4] + epsilon)
    {
        v.z *= damping;
        particle.position.z = bound[4] + epsilon + 0.3 * noise(particle.position);
    }
    else if(particles[id.x].position.z > bound[5] - epsilon) 
    {
        v.z *= damping;
        particle.position.z = bound[5] - epsilon - 0.3 * noise(particle.position);
    }
    velocity[id.x] = v;
    particles[id.x] = particle;
}

#include "../Noise/noise.cginc"
#pragma kernel noise_density
static const int n_thread = 100;
RWStructuredBuffer<float3> points;
RWStructuredBuffer<float> noise_densities;

[numthreads(n_thread, 1, 1)]
void noise_density(int3 id : SV_DispatchThreadID)
{
    float3 pos = particles[id.x].position;
    float noise = snoise(pos),
    density_value = -pos.y + noise;
    points[id.x] = pos;
    noise_densities[id.x] = density_value;
}

#include "march_table.compute"
#pragma kernel march

struct tri
{
    float3 vertex_a,
    vertex_b,
    vertex_c;
};

AppendStructuredBuffer<tri> triangles;
float isolevel;
static const int num_thread = 4,
n_point_per_axis;
RWStructuredBuffer<int> int_debug;
RWStructuredBuffer<float> float_debug;

int hash(int x, int y, int z)
{
    return x + n_point_per_axis * (y + n_point_per_axis * z);
}

int map(int x, int y, int z)
{
    return (x * n_thread + y) * n_thread + z; /* x * n_thread * n_thread + y * n_thread + z */
}

float3 interpolate_vertex(float3 vertex1, float density1, float3 vertex2, float density2)
{
    float t = (isolevel - density1) / (density2 - density1);
    return vertex1 + t * (vertex2 - vertex1);
}

[numthreads(num_thread, num_thread, num_thread)]
void march(int3 id : SV_DispatchThreadID)
{
    if(id.x >= n_point_per_axis - 1 || id.y >= n_point_per_axis - 1 || id.z >= n_point_per_axis - 1) /* voxel includes neighbor points */
        return;
    float3 cube_corner_vertices[] =
    {
        points[hash(id.x, id.y, id.z)],
        points[hash(id.x + 1, id.y, id.z)],
        points[hash(id.x + 1, id.y, id.z + 1)],
        points[hash(id.x, id.y, id.z + 1)],
        points[hash(id.x, id.y + 1, id.z)],
        points[hash(id.x + 1, id.y + 1, id.z)],
        points[hash(id.x + 1, id.y + 1, id.z + 1)],
        points[hash(id.x, id.y + 1, id.z + 1)]
    };
    float cube_corner_densities[] =
    {
        noise_densities[hash(id.x, id.y, id.z)],
        noise_densities[hash(id.x + 1, id.y, id.z)],
        noise_densities[hash(id.x + 1, id.y, id.z + 1)],
        noise_densities[hash(id.x, id.y, id.z + 1)],
        noise_densities[hash(id.x, id.y + 1, id.z)],
        noise_densities[hash(id.x + 1, id.y + 1, id.z)],
        noise_densities[hash(id.x + 1, id.y + 1, id.z + 1)],
        noise_densities[hash(id.x, id.y + 1, id.z + 1)]
    };
    uint n_corner = 8,
    cube_index = 0;
    for(uint c = 0; c < n_corner; ++c)
        if(cube_corner_densities[c] < isolevel) /* if the input data is not binary, need an additional parameter (threshold value or isovalue) to classify samples as inside or outside the surface */
            cube_index |= 1 << c;
    int_debug[map(id.x, id.y, id.z)] = id.x;
    for(uint i = 0; triangle_table[cube_index][i] != -1; i += 3)
    {
        int a1 = corner_index_a_from_edge[triangle_table[cube_index][i]],
        b1 = corner_index_b_from_edge[triangle_table[cube_index][i]],
        a2 = corner_index_a_from_edge[triangle_table[cube_index][i + 1]],
        b2 = corner_index_b_from_edge[triangle_table[cube_index][i + 1]],
        a3 = corner_index_a_from_edge[triangle_table[cube_index][i + 2]],
        b3 = corner_index_b_from_edge[triangle_table[cube_index][i + 2]];
        tri a;
        a.vertex_a = interpolate_vertex(cube_corner_vertices[a1], cube_corner_densities[a1], cube_corner_vertices[b1], cube_corner_densities[b1]);
        a.vertex_b = interpolate_vertex(cube_corner_vertices[a2], cube_corner_densities[a2], cube_corner_vertices[b2], cube_corner_densities[b2]);
        a.vertex_c = interpolate_vertex(cube_corner_vertices[a3], cube_corner_densities[a3], cube_corner_vertices[b3], cube_corner_densities[b3]);
        triangles.Append(a);
    }
}

#pragma kernel NoiseFieldGenerator
uint3 Dims;
float Scale;
float Time;
RWStructuredBuffer<float> Voxels;

[numthreads(8, 8, 8)]
void NoiseFieldGenerator(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x + Dims.x * (id.y + Dims.y * id.z);
    float3 pos = particles[i].position;
    float noise = snoise(pos),
    density_value = -pos.y + noise;
    Voxels[i] = density_value;
}

/* libConst.cginc */
#ifndef __const__
#define __const__

#define e  		2.7182818
#define HALF_MAX        65504.0 // (2 - 2^-10) * 2^15
#define HALF_MAX_MINUS1 65472.0 // (2 - 2^-9) * 2^15
#define EPSILON         1.0e-5
#define PI              3.14159265359
#define TWO_PI          6.28318530718
#define FOUR_PI         12.56637061436
#define INV_PI          0.31830988618
#define INV_TWO_PI      0.15915494309
#define INV_FOUR_PI     0.07957747155
#define HALF_PI         1.57079632679
#define INV_HALF_PI     0.636619772367

//compute shader features
#define MAX_THREAD_Z	64

#ifdef SHADER_API_METAL  //ios or mac 
#define MAX_THREAD		512
#define MAX_THREAD_X	512
#define MAX_THREAD_Y	512
#define REV_THREAD_Z    8  // as MAX_THREAD/MAX_THREAD_Z
#define THREAD_Y_32Z    16 //MAX_THREAD/32
#define THREAD_Y_64Z    8
#define THREAD_Y_128Z   4
#define THREAD_Y_256Z   2
#else
#define MAX_THREAD		1024
#define MAX_THREAD_X	1024
#define MAX_THREAD_Y	1024
#define REV_THREAD_Z    16  // as 1024/64
#define THREAD_Y_32Z    32
#define THREAD_Y_64Z    16
#define THREAD_Y_128Z   8
#define THREAD_Y_256Z   4
#endif

#define MAX_GROUP_SHARED	8192 //globalshared's count max is 8192's float (equal 32768bytes)

#define FLT_EPSILON     1.192092896e-07 // Smallest positive number, such that 1.0 + FLT_EPSILON != 1.0
#define FLT_MIN         1.175494351e-38 // Minimum representable positive floating-point number
#define FLT_MAX         3.402823466e+38 // Maximum representable floating-point number

#endif

/* libStd.cginc */
#ifndef __std__
#define __std__

/* used in encoder & decoder */

#define CACHE_MAX  2048
#define CACHE_HALF 1024

groupshared float g_cache[CACHE_MAX];


#define StdIndex(x, y, z, width, depth) \
	((width) * (depth) * (x) + (depth) * (y) + z)


#define StdID(id, width, depth)	\
	((width) * (depth) * id.x + (depth) * id.y + id.z)


#define StdPad(id, width, depth, pad)	\
	uint low = pad - id.x;	\
	uint mid = id.x - pad;	\
	uint high = 2 * width + pad - 1 - id.x;	\
	uint x_array[3] = { low, mid, high };	\
	low = pad - id.y;	\
	mid = id.y - pad;	\
	high = 2 * width + pad - 1 - id.y;	\
	uint y_array[3] = { low, mid, high };	\
	uint x_id = id.x >= (pad + width) ? 2 : saturate(id.x / pad);	\
	uint y_id = id.y >= (pad + width) ? 2 : saturate(id.y / pad);	\
	x_id = x_array[x_id];	\
	y_id = y_array[y_id];	\
	uint indx = StdIndex(x_id, y_id, id.z, width, depth);	\
	uint indx2 = StdID(id, width + pad * 2, depth);	\


void StdSeq(int x, int y, int z, int width,int depth, out int res[9])
{
	res[0] = StdIndex(x,	y,	 z,	width,	depth);
	res[1] = StdIndex(x+1,	y,	 z,	width,	depth);
	res[2] = StdIndex(x+2,	y,	 z,	width,	depth);
	res[3] = StdIndex(x,	y+1, z,	width,	depth);
	res[4] = StdIndex(x+1,	y+1, z,	width,	depth);
	res[5] = StdIndex(x+2,	y+1, z,	width,	depth);
	res[6] = StdIndex(x,	y+2, z,	width,	depth);
	res[7] = StdIndex(x+1,	y+2, z,	width,	depth);
	res[8] = StdIndex(x+2,  y+2, z,	width,	depth);
}


/*
conv2d with valid padding
*/
float3x3 StdSample(RWStructuredBuffer<float> buffer, int x, int y, int z, int width, int depth)
{
	int sq[9];
	StdSeq(x, y, z, width, depth, sq);
	return float3x3(buffer[sq[0]], buffer[sq[1]], buffer[sq[2]],
		buffer[sq[3]], buffer[sq[4]], buffer[sq[5]],
		buffer[sq[6]], buffer[sq[7]], buffer[sq[8]]);
}

/*
conv2d with same padding 
xy range section will be filled with zero
*/
float3x3 StdSlowSample(RWStructuredBuffer<float> buffer, int x, int y, int z, int width, int depth)
{
	float a[9];
	bool x1 = x + 1 < width;
	bool x2 = x + 2 < width;
	bool y1 = y + 1 < width;
	bool y2 = y + 2 < width;
	a[0] = buffer[StdIndex(x, y, z, width, depth)];
	a[1] = x1 ? buffer[StdIndex(x + 1, y, z, width, depth)] : 0;
	a[2] = x2 ? buffer[StdIndex(x + 2, y, z, width, depth)] : 0;
	a[3] = y1 ? buffer[StdIndex(x, y + 1, z, width, depth)] : 0;
	a[4] = x1 && y1 ? buffer[StdIndex(x + 1, y + 1, z, width, depth)] : 0;
	a[5] = x2 && y1 ? buffer[StdIndex(x + 2, y + 1, z, width, depth)] : 0;
	a[6] = y2 ? buffer[StdIndex(x, y + 2, z, width, depth)] : 0;
	a[7] = x1 && y2 ? buffer[StdIndex(x + 1, y + 2, z, width, depth)] : 0;
	a[8] = x2 && y2 ? buffer[StdIndex(x + 2, y + 2, z, width, depth)] : 0;
	return float3x3(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}


inline bool StdCheckRange(uint3 id, uint width)
{
	return id.x >= width || id.y >= width;
}

#define StdInnerNormal(inbuffer)	\
	[unroll]	\
	for (uint j = 0; j < nwidth; j++) {	\
		for (uint i = 0; i < intvl; i++) {	\
			int idx = j * nwidth * depth + (id.y + width * i) * depth + z;	\
			g_cache[nix] += inbuffer[idx];	\
			g_cache[nix + offset] += inbuffer[idx] * inbuffer[idx];	\
		}	\
	}


#define StdDefineNormal(id, inbuffer, outbuffer, width)	\
	uint offset = CACHE_HALF;	\
	uint z = id.x;	\
	uint nix = id.y * depth + z;	\
	uint scale = nwidth / width;	\
	uint intvl = id.y < nwidth % width ?  scale + 1 : scale;	\
	g_cache[nix] = 0;	\
	g_cache[nix + offset] = 0;	\
	StdInnerNormal(inbuffer)	\
	GroupMemoryBarrierWithGroupSync();	\
	if (id.y == 1)	\
	{	\
		float mean = 0, qrt = 0;	\
		for (uint i = 0; i < width; i++)	\
		{	\
			int idx = i * depth + z;	\
			mean += g_cache[idx];	\
			qrt += g_cache[idx + offset];	\
		}	\
		int len = nwidth * nwidth;	\
		mean = mean / len;	\
		outbuffer[z * 2] = mean;	\
		outbuffer[z * 2 + 1] = qrt / len - mean * mean;	\
	}	

#endif

/* libActive.cginc */
#ifndef __active__
#define __active__

#include "libConst.cginc"

inline float relu(float x)
{
	return max(x,0);
}

inline half relu(half x)
{
	return max(x, 0);
}

inline float3 relu(float3 x)
{
	float x1 = relu(x.x);
	float x2 = relu(x.y);
	float x3 = relu(x.z);
	return float3(x1, x2, x3);
}

inline half3 relu(half3 x)
{
	return half3(relu(x.x), relu(x.y), relu(x.z));
}

inline float lrelu(float x,float leak)
{
	return max(x, 0) + leak * min(x, 0);
}

inline half lrelu(half x, half leak)
{
	return max(x, 0) + leak * min(x, 0);
}

inline float3 lrelu(float3 x, float leak)
{
	return float3(lrelu(x.x, leak), lrelu(x.y, leak), lrelu(x.z, leak));
}

inline half3 lrelu(half3 x, half leak)
{
	return half3(lrelu(x.x, leak), lrelu(x.y, leak), lrelu(x.z, leak));
}

inline float lrelu(float x)
{
	return lrelu(x,0.2);
}

inline half lrelu(half x)
{
	return lrelu(x, 0.2);
}

inline float3 lrelu(float3 x)
{
	return lrelu(x, 0.2);
}

inline half3 lrelu(half3 x)
{
	return lrelu(x, 0.2);
}

inline float sigmod(float x)
{
	return 1/(1 + pow(e,-x));
}

inline half sigmod(half x)
{
	return 1 / (1 + pow(e, -x));
}

inline float3 sigmod(float3 x)
{
	return float3(sigmod(x.x), sigmod(x.y), sigmod(x.z));
}

inline half3 sigmod(half3 x)
{
	return half3(sigmod(x.x), sigmod(x.y), sigmod(x.z));
}

#endif

/* libEncoder.cginc */
#ifndef __encoder__
#define __encoder__


#include "libActive.cginc"
#include "libStd.cginc"


#define DefineEncodeBuffer(seq)	\
	RWStructuredBuffer<float> encoder_conv##seq##;	\
	RWStructuredBuffer<float> encoder_conv##seq##_statistic;	\


#define DefineEncoderArg(seq)	\
	StructuredBuffer<float> encoder_g_e##seq##_bn_offset;	\
	StructuredBuffer<float> encoder_g_e##seq##_bn_scale;	\
	StructuredBuffer<float3x3> encoder_g_e##seq##_c_Conv_weights;	\


#define DefineEncoderConv(id, input, output, depth1, depth2, stride, idx, pidx)	\
	for(uint j = 0; j < depth2; j++)	\
	{	\
		float v = 0.0f;	\
		for(uint i = 0; i < depth1; i++)	\
		{	\
			float3x3 xsam = StdSample(encoder_conv##pidx##, id.x*stride, id.y*stride, i, input, depth1);	\
			float3x3 conv = encoder_g_e##idx##_c_Conv_weights[depth2 * i + j];	\
			float3x3 imul = xsam * conv;	\
			float3 iall = imul[0] + imul[1]+ imul[2];	\
			v += iall[0] + iall[1] + iall[2];	\
		}	\
		int indx = (output * depth2) * id.x  + depth2 * id.y  + j;	\
		encoder_conv##idx##[indx] = v;	\
	}


#define DefineEnInstRelu(id, width, depth, seq)	\
	int inx = StdID(id, width, depth);	\
	float color = encoder_conv##seq##[inx];	\
	float mean = encoder_conv##seq##_statistic[id.z * 2];	\
	float variance = encoder_conv##seq##_statistic[id.z * 2 + 1];	\
	float inv = rsqrt(variance + EPSILON);	\
	float normalized = (color - mean) * inv;	\
	float scale = encoder_g_e##seq##_bn_scale[id.z];	\
	float offset = encoder_g_e##seq##_bn_offset[id.z];	\
	encoder_conv##seq##[inx] = relu(scale * normalized + offset);
    
#endif

/* libDecoderArgs.cginc */
#ifndef __decoder_arg__
#define __decoder_arg__

//network args 
StructuredBuffer<float3x3> decoder_g_d1_dc_conv2d_Conv_weights;
StructuredBuffer<float3x3> decoder_g_d2_dc_conv2d_Conv_weights;
StructuredBuffer<float3x3> decoder_g_d3_dc_conv2d_Conv_weights;
StructuredBuffer<float3x3> decoder_g_d4_dc_conv2d_Conv_weights;

StructuredBuffer<float3x3> decoder_g_r1_c1_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r1_c2_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r2_c1_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r2_c2_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r3_c1_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r3_c2_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r4_c1_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r4_c2_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r5_c1_Conv_weights;
StructuredBuffer<float3x3> decoder_g_r5_c2_Conv_weights;

StructuredBuffer<float> decoder_g_pred_c_Conv_weights; //7x7

StructuredBuffer<float> decoder_g_d1_bn_offset;
StructuredBuffer<float> decoder_g_d1_bn_scale;
StructuredBuffer<float> decoder_g_d2_bn_offset;
StructuredBuffer<float> decoder_g_d2_bn_scale;
StructuredBuffer<float> decoder_g_d3_bn_offset;
StructuredBuffer<float> decoder_g_d3_bn_scale;
StructuredBuffer<float> decoder_g_d4_bn_offset;
StructuredBuffer<float> decoder_g_d4_bn_scale;

StructuredBuffer<float> decoder_g_r1_bn1_offset;
StructuredBuffer<float> decoder_g_r1_bn1_scale;
StructuredBuffer<float> decoder_g_r1_bn2_offset;
StructuredBuffer<float> decoder_g_r1_bn2_scale;
StructuredBuffer<float> decoder_g_r2_bn1_offset;
StructuredBuffer<float> decoder_g_r2_bn1_scale;
StructuredBuffer<float> decoder_g_r2_bn2_offset;
StructuredBuffer<float> decoder_g_r2_bn2_scale;
StructuredBuffer<float> decoder_g_r3_bn1_offset;
StructuredBuffer<float> decoder_g_r3_bn1_scale;
StructuredBuffer<float> decoder_g_r3_bn2_offset;
StructuredBuffer<float> decoder_g_r3_bn2_scale;
StructuredBuffer<float> decoder_g_r4_bn1_offset;
StructuredBuffer<float> decoder_g_r4_bn1_scale;
StructuredBuffer<float> decoder_g_r4_bn2_offset;
StructuredBuffer<float> decoder_g_r4_bn2_scale;
StructuredBuffer<float> decoder_g_r5_bn1_offset;
StructuredBuffer<float> decoder_g_r5_bn1_scale;
StructuredBuffer<float> decoder_g_r5_bn2_offset;
StructuredBuffer<float> decoder_g_r5_bn2_scale;

#endif

/* libDecoder.cginc */
#ifndef __decoder__
#define __decoder__

#include "libConst.cginc"
#include "libStd.cginc"
#include "libActive.cginc"

#define DefineDecodeBuffer(seq)	\
	RWStructuredBuffer<float> decoder_conv##seq##;	\
	RWStructuredBuffer<float> decoder_conv##seq##_conved;		\
	RWStructuredBuffer<float> decoder_conv##seq##_statistic;	\


#define DefineResiduleConv(id, input, output, seq, idx) \
	for(int j = 0;j < depth; j++) \
	{ 	\
		float v = 0.0f;	\
		for(int i= 0; i < depth; i++)	\
		{	\
			float3x3 xsamp = StdSample(decoder_conv0, id.x, id.y, i, input, depth);	\
			float3x3 kernel = decoder_g_r##seq##_c##idx##_Conv_weights[depth * i + j];	\
			float3x3 conv = xsamp * kernel;	\
			float3 iall = conv[0] + conv[1] + conv[2];	\
			v += iall[0] + iall[1] + iall[2];	\
		}	\
		int indx = (output * depth) * id.x + depth * id.y + j;	\
		input_writable[indx] = v;	\
	}


#define DefineResiduleInst(id, seq, r)	\
	int indx = StdID(id, width, depth);	\
	float color = input_writable[indx];	\
	float mean = decoder_conv0_statistic[id.z * 2];	\
	float variance = decoder_conv0_statistic[id.z * 2 + 1];	\
	float inv = rsqrt(variance + EPSILON);	\
	float normalized = (color - mean) * inv;	\
	float scale = decoder_g_r##seq##_bn##r##_scale[id.z];	\
	float offset = decoder_g_r##seq##_bn##r##_offset[id.z];	\
	input_writable[indx] = scale * normalized + offset;		


#define DefineDecoderConv(id, width, depth1, depth2, idx)	\
	for(uint j = 0; j < depth2; j++) \
	{ 	\
		float v = 0.0f;	\
		for(uint i = 0; i < depth1; i++)	\
		{	\
			float3x3 xsamp = StdSlowSample(decoder_conv##idx##_conved, id.x, id.y, i, width, depth1);	\
			float3x3 kernel = decoder_g_d##idx##_dc_conv2d_Conv_weights[depth2 * i + j];	\
			float3x3 conv = xsamp * kernel;	\
			float3 iall = conv[0] + conv[1] + conv[2];	\
			v += iall[0] + iall[1] + iall[2];	\
		}	\
		int indx = (width * depth2) * id.x + depth2 * id.y + j;	\
		decoder_conv##idx##[indx] = v;	\
	}


#define DefineDecoderInstRelu(id, seq)	\
	int inx = StdID(id, width, depth);	\
	float color = decoder_conv##seq##[inx];	\
	float mean = decoder_conv##seq##_statistic[id.z * 2];	\
	float variance = decoder_conv##seq##_statistic[id.z * 2 + 1];	\
	float inv = rsqrt(variance + EPSILON);	\
	float normalized = (color - mean) * inv;	\
	float scale = decoder_g_d##seq##_bn_scale[id.z];	\
	float offset = decoder_g_d##seq##_bn_offset[id.z];	\
	decoder_conv##seq##[inx] = relu(scale * normalized + offset);


#define DefineDecoderPad(id, idx)	\
	int ninx1 = (2 * width) * depth * (2 * id.x) + depth * (2* id.y) + id.z;	\
	int ninx2 = (2 * width) * depth * (2 * id.x) + depth * (2* id.y + 1) + id.z;	\
	int ninx3 = (2 * width) * depth * (2 * id.x+1) + depth * (2* id.y) + id.z;	\
	int ninx4 = (2 * width) * depth * (2 * id.x+1) + depth * (2* id.y + 1) + id.z;	\
	decoder_conv##idx##_conved[ninx1] = v;	\
	decoder_conv##idx##_conved[ninx2] = v;	\
	decoder_conv##idx##_conved[ninx3] = v;	\
	decoder_conv##idx##_conved[ninx4] = v;	\


#define DefineDecoderExpand(id, idx, pidx) \
	int indx = StdID(id, width, depth);	\
	float v = decoder_conv##pidx##[indx];	\
	DefineDecoderPad(id, idx)

#endif

/* StyleEncoder.compute */
#pragma kernel StyleConv0
#pragma kernel StyleNormal0
#pragma kernel StyleInstance0
#pragma kernel StylePad
#pragma kernel StyleConv1
#pragma kernel StyleNormal1
#pragma kernel StyleInstance1
#pragma kernel StyleConv2
#pragma kernel StyleNormal2
#pragma kernel StyleInstance2
#pragma kernel StyleConv3
#pragma kernel StyleNormal3 
#pragma kernel StyleInstance3 
#pragma kernel StyleConv4
#pragma kernel StyleNormal4 
#pragma kernel StyleInstance4
#pragma kernel StyleConv5
#pragma kernel StyleNormal5
#pragma kernel StyleInstance5

#include "libEncoder.cginc"

uniform int alpha;
Texture2D<float4> source;
RWStructuredBuffer<float> encoder_inst;
StructuredBuffer<float> encoder_g_e0_bn_offset;
StructuredBuffer<float> encoder_g_e0_bn_scale;
DefineEncoderArg(1)
DefineEncoderArg(2)
DefineEncoderArg(3)
DefineEncoderArg(4)
DefineEncoderArg(5)
DefineEncodeBuffer(0)
DefineEncodeBuffer(1)
DefineEncodeBuffer(2)
DefineEncodeBuffer(3)
DefineEncodeBuffer(4)
DefineEncodeBuffer(5)


/*
***  formula  o=(w-k+2p)/s+1  ***
encoder construct as:
init  256x256x3->
pad	 	286x286x3->  
conv1	  284x284x32->
conv2		141x141x32->
conv3		  70x70x64->
conv4			34x34x128->
conv5			  16x16x256		
*/

[numthreads(8, 8, 3)]
void StyleConv0(uint3 id : SV_DispatchThreadID)
{
	uint width = 256, depth = 3;
	float4 color = source[uint2(id.x * alpha, alpha * (width - 1 - id.y))];
	float arr[3] = { color.x, color.y, color.z };
	float v = 2 * arr[id.z] - 1;  // (0,1) -> (-1,1)
	int idx = id.y * width * depth + id.x * depth + id.z;
	encoder_inst[idx] = v;
}

[numthreads(3, 64, 1)]
void StyleNormal0(uint3 id : SV_DispatchThreadID) // 256x256x3
{
	uint width = 64, depth = 3, nwidth = 256;
	StdDefineNormal(id, encoder_inst, encoder_conv0_statistic, width);
}

[numthreads(8, 8, 3)]
void StyleInstance0(uint3 id : SV_DispatchThreadID) //id.xy=256 256x256x3
{
	uint width = 256, depth = 3;
	int idx = StdID(id, width, depth);
	float color = encoder_inst[idx];
	float mean = encoder_conv0_statistic[id.z * 2];
	float variance = encoder_conv0_statistic[id.z * 2 + 1];
	float inv = rsqrt(variance + EPSILON);
	float normalized = (color - mean) * inv;
	float scale = encoder_g_e0_bn_scale[id.z];
	float offset = encoder_g_e0_bn_offset[id.z];
	encoder_inst[idx] = scale * normalized + offset;
}

[numthreads(8, 8, 3)]
void StylePad(uint3 id : SV_DispatchThreadID) //id.xy=288 256x256x3->286x286x3
{
	uint pad = 15, width = 256, depth = 3;
	if (StdCheckRange(id, width + 2 * pad)) return;
	StdPad(id, width, depth, pad)
	encoder_conv0[indx2] = encoder_inst[indx];
}

[numthreads(8,8,1)]
void StyleConv1 (uint3 id : SV_DispatchThreadID) //id.xy=288 286x286x3->284x284x32
{
	uint input = 286, output = 284, depth1 = 3, stride =1;
	uint depth2 = StdCheckRange(id, output) ? 0 : 32;
	DefineEncoderConv(id, input, output, depth1, depth2, stride, 1, 0);
}

[numthreads(32, THREAD_Y_32Z, 1)]
void StyleNormal1(uint3 id : SV_DispatchThreadID) //284x284x32
{
	uint width = THREAD_Y_32Z, depth = 32, nwidth = 284;
	StdDefineNormal(id, encoder_conv1, encoder_conv1_statistic, width);
}

[numthreads(8, 8, 4)]
void StyleInstance1(uint3 id : SV_DispatchThreadID) //284x284x32
{
	uint width = 284, depth = 32;
	if (StdCheckRange(id, width)) return;
	DefineEnInstRelu(id, width, depth, 1);
}

[numthreads(8,8,1)]
void StyleConv2(uint3 id : SV_DispatchThreadID) //id.xy=144 284x284x32->141x141x32 
{
	uint input = 284, output = 141, depth1 = 32, stride = 2;
	uint depth2 = StdCheckRange(id, output) ? 0 : 32;
	DefineEncoderConv(id, input, output, depth1, depth2, stride, 2, 1);
}

[numthreads(32,THREAD_Y_32Z,1)]
void StyleNormal2(uint3 id : SV_DispatchThreadID)  //141x141x32 
{
	uint width = THREAD_Y_32Z, depth = 32, nwidth = 141;
	StdDefineNormal(id, encoder_conv2, encoder_conv2_statistic, width);
}

[numthreads(8,8,4)]
void StyleInstance2(uint3 id : SV_DispatchThreadID) //id.xy=144 141x141x32
{
	uint width = 141, depth = 32;
	if (StdCheckRange(id, width)) return;
	DefineEnInstRelu(id, width, depth, 2);
}

[numthreads(8,8,1)]
void StyleConv3 (uint3 id : SV_DispatchThreadID) //id.xy=72 141x141x32->70x70x64
{
	uint input = 141, output = 70, depth1 = 32, stride =2;
	uint depth2 = StdCheckRange(id, output) ? 0 : 64;
	DefineEncoderConv(id, input, output, depth1, depth2, stride, 3, 2)
}

[numthreads(MAX_THREAD_Z,REV_THREAD_Z,1)]
void StyleNormal3(uint3 id : SV_DispatchThreadID)  // 70x70x64
{
	uint width = REV_THREAD_Z, depth = 64, nwidth = 70;
	StdDefineNormal(id, encoder_conv3, encoder_conv3_statistic, width);
}

[numthreads(8,8,4)]
void StyleInstance3 (uint3 id : SV_DispatchThreadID) //id.xy=72 70x70x64 
{
	uint width = 70, depth = 64;
	if (StdCheckRange(id, width)) return;
	DefineEnInstRelu(id, width, depth, 3);
}

[numthreads(8,8,1)]
void StyleConv4 (uint3 id : SV_DispatchThreadID) //id.xy=40 70x70x64->34x34x128
{
	uint input = 70, output = 34, depth1 = 64, stride =2;
	uint depth2 = StdCheckRange(id, output) ? 0 : 128;
	DefineEncoderConv(id, input, output, depth1, depth2, stride, 4, 3);
}

[numthreads(128, THREAD_Y_128Z, 1)] 
void StyleNormal4(uint3 id : SV_DispatchThreadID) //34x34x128
{
	uint width = THREAD_Y_128Z, depth = 128, nwidth = 34;
	StdDefineNormal(id, encoder_conv4, encoder_conv4_statistic, width);	
}

[numthreads(8,8,4)]
void StyleInstance4(uint3 id : SV_DispatchThreadID) //id.xy=40 34x34x128
{
	uint width = 34, depth = 128;
	if (StdCheckRange(id, width)) return;
	DefineEnInstRelu(id, width, depth, 4);
}

[numthreads(8,8,1)]
void StyleConv5(uint3 id : SV_DispatchThreadID) //id.xy=40 34x34x128->16x16x256
{
	uint input = 34, output = 16, depth1 = 128, stride = 2;
	uint depth2 = StdCheckRange(id, output) ? 0 : 256;
	DefineEncoderConv(id, input, output, depth1, depth2, stride, 5, 4);
}

[numthreads(256, THREAD_Y_256Z, 1)]
void StyleNormal5(uint3 id : SV_DispatchThreadID)  //16x16x256
{
	uint width = THREAD_Y_256Z, depth = 256, nwidth = 16;
	StdDefineNormal(id, encoder_conv5, encoder_conv5_statistic, width);
}

[numthreads(8,8,4)]
void StyleInstance5 (uint3 id : SV_DispatchThreadID) //id.xy=16 16x16x256
{
	uint width = 16, depth = 256;
	if (StdCheckRange(id, width)) return;
	DefineEnInstRelu(id, width, depth, 5);
}

/* StyleDecoder.compute */
#pragma kernel ResidulePad1_1
#pragma kernel ResiduleConv1_1
#pragma kernel ResiduleNormal1_1
#pragma kernel ResiduleInst1_1
#pragma kernel ResidulePad1_2
#pragma kernel ResiduleConv1_2
#pragma kernel ResiduleNormal1_2
#pragma kernel ResiduleInst1_2
#pragma kernel DecoderExpand1
#pragma kernel DecoderConv1
#pragma kernel DecoderNormal1
#pragma kernel DecoderInstance1
#pragma kernel DecoderExpand2
#pragma kernel DecoderConv2
#pragma kernel DecoderNormal2
#pragma kernel DecoderInstance2
#pragma kernel DecoderExpand3
#pragma kernel DecoderConv3
#pragma kernel DecoderNormal3
#pragma kernel DecoderInstance3
#pragma kernel DecoderExpand4
#pragma kernel DecoderConv4
#pragma kernel DecoderNormal4
#pragma kernel DecoderInstance4
#pragma kernel DecoderPad5
#pragma kernel DecoderConv5

RWStructuredBuffer<float> input_initial;
RWStructuredBuffer<float> input_writable;
RWTexture2D<float4> decoder_destination;

#include "libStd.cginc"
#include "libDecoderArgs.cginc"
#include "libDecoder.cginc"

DefineDecodeBuffer(0)
DefineDecodeBuffer(1)
DefineDecodeBuffer(2)
DefineDecodeBuffer(3)
DefineDecodeBuffer(4)
RWStructuredBuffer<float> decoder_conv5_pad;		// 262x262x32

/*
***  formula  o=(w-k+2p)/s+1  ***
encoder construct as:
init  16x16x256->
resid	16x16x256->  
decv1	  32x32x256->
decv2		64x64x128->
decv3		  128x128x64->
decv4			256x256x32->
pad 				262x262x32->
conv(pred)				256x256x3		
*/

/*
residule-block
16x16x256->18x18x256->16x16x256
*/
[numthreads(8,8,4)]
void ResidulePad1_1(uint3 id : SV_DispatchThreadID) //id.xy=24  18x18x256
{
	uint pad = 1, width = 16, depth = 256;
	if (StdCheckRange(id, width + 2 * pad)) return;
	StdPad(id, width, depth, pad);
	decoder_conv0[indx2] = input_initial[indx];
}

[numthreads(8,8,1)]
void ResiduleConv1_1(uint3 id: SV_DispatchThreadID) //id.xy=16 18x18x256->16x16x256
{
	int input = 18, output = 16, depth = 256;
	DefineResiduleConv(id, input, output, 1, 1)
}

[numthreads(256,THREAD_Y_256Z,1)]
void ResiduleNormal1_1(uint3 id: SV_DispatchThreadID) //id.z=256 16x16x256
{
	uint width = THREAD_Y_256Z, depth = 256, nwidth = 16;
	StdDefineNormal(id, input_writable, decoder_conv0_statistic, width);
}

[numthreads(8,8,4)]
void ResiduleInst1_1(uint3 id:SV_DispatchThreadID) //id.xy=16 16x16x256
{
	uint width = 16, depth = 256;
	DefineResiduleInst(id, 1, 1);
	input_writable[indx] = input_writable[indx];
}

[numthreads(8, 8, 4)]
void ResidulePad1_2(uint3 id : SV_DispatchThreadID) //id.xy=24  18x18x256
{
	uint pad = 1, width = 16, depth = 256;
	if (StdCheckRange(id, width + 2 * pad)) return;
	StdPad(id, width, depth, pad);
	decoder_conv0[indx2] = relu(input_writable[indx]);
}

[numthreads(8,8,1)]
void ResiduleConv1_2(uint3 id: SV_DispatchThreadID) //id.xy=16 18x18x256->16x16x256
{
	int input = 18, output = 16, depth = 256;
	DefineResiduleConv(id, input, output, 1, 2)
}

[numthreads(256,THREAD_Y_256Z,1)]
void ResiduleNormal1_2(uint3 id: SV_DispatchThreadID) //id.xy=16 16x16x256
{
	uint width = 4, depth = 256, nwidth = 16;
	StdDefineNormal(id, input_writable, decoder_conv0_statistic, width);
}

[numthreads(8,8,4)]
void ResiduleInst1_2(uint3 id:SV_DispatchThreadID) //id.xy=16 16x16x256
{
	uint width = 16, depth = 256;
	DefineResiduleInst(id, 1, 2);
	input_writable[indx] += input_initial[indx];
}

[numthreads(8,8,4)]
void DecoderExpand1(uint3 id: SV_DispatchThreadID) //id.xy=16 16x16x256->32x32x256
{
	int width = 16, depth = 256;
	int indx = StdID(id, width, depth);
	float v = input_writable[indx];
	DefineDecoderPad(id, 1)
}

[numthreads(8,8,1)]
void DecoderConv1(uint3 id: SV_DispatchThreadID) //id.xy=32 32x32x256->32x32x256
{
	uint width = 32, depth1 = 256, depth2 = 256;
	DefineDecoderConv(id, width, depth1, depth2, 1);
}

[numthreads(256, THREAD_Y_256Z, 1)]
void DecoderNormal1(uint3 id: SV_DispatchThreadID) //id.z=256 32x32x256
{
	uint width = THREAD_Y_256Z, depth = 256, nwidth = 32;
	StdDefineNormal(id, decoder_conv1, decoder_conv1_statistic, width);
}

[numthreads(8,8,4)]
void DecoderInstance1(uint3 id: SV_DispatchThreadID) //id.xy=32 32x32x256
{
	uint width = 32, depth = 256;
	DefineDecoderInstRelu(id, 1)
}

[numthreads(8,8,4)]
void DecoderExpand2(uint3 id: SV_DispatchThreadID) //id.xy=32, 32x32x256->64x64x256
{
	int width = 32, depth = 256;
	DefineDecoderExpand(id, 2, 1);
}

[numthreads(8,8,1)]
void DecoderConv2(uint3 id: SV_DispatchThreadID) //id.xy=64 64x64x256->64x64x128
{
	uint width = 64, depth1 = 256, depth2 = 128;
	DefineDecoderConv(id, width, depth1, depth2, 2);
}

[numthreads(128, THREAD_Y_128Z, 1)]
void DecoderNormal2(uint3 id: SV_DispatchThreadID) //id.z=MAX_THREAD_Z*2 64x64x128
{
	uint width = THREAD_Y_128Z, depth = 128, nwidth = 64;
	StdDefineNormal(id, decoder_conv2, decoder_conv2_statistic, width);
}

[numthreads(8,8,4)]
void DecoderInstance2(uint3 id: SV_DispatchThreadID) //id.xy=64 64x64x128
{
	uint width = 64, depth = 128;
	DefineDecoderInstRelu(id, 2)
}

[numthreads(8,8,4)]
void DecoderExpand3(uint3 id: SV_DispatchThreadID)//id.xy=64 64x64x128->128x128x128 
{
	int width = 64, depth = 128;
	DefineDecoderExpand(id, 3, 2);
}

[numthreads(8,8,1)]
void DecoderConv3(uint3 id: SV_DispatchThreadID) //id.xy=128 128x128x128->128x128x64
{
	uint width = 128, depth1 = 128, depth2 = 64;
	DefineDecoderConv(id, width, depth1, depth2, 3);
}

[numthreads(MAX_THREAD_Z, REV_THREAD_Z, 1)]
void DecoderNormal3(uint3 id: SV_DispatchThreadID) //128x128x64
{
	uint width = REV_THREAD_Z, depth = 64, nwidth = 128;
	StdDefineNormal(id, decoder_conv3, decoder_conv3_statistic, width);
}

[numthreads(8,8,4)]
void DecoderInstance3(uint3 id: SV_DispatchThreadID) //id.xy=128 128x128x64
{
	uint width = 128, depth = 64;
	DefineDecoderInstRelu(id, 3)
}

[numthreads(8,8,4)]
void DecoderExpand4(uint3 id: SV_DispatchThreadID) //id.xy=128 128x128x64->256x256x64
{
	int width = 128, depth = 64;
	DefineDecoderExpand(id, 4, 3);
}

[numthreads(8,8,1)]
void DecoderConv4(uint3 id: SV_DispatchThreadID) //id.xy=256 256x256x64->256x256x32
{
	uint width = 256, depth1 = 64, depth2 = 32;
	DefineDecoderConv(id, width, depth1, depth2, 4);
}

[numthreads(32, THREAD_Y_32Z, 1)]
void DecoderNormal4(uint3 id: SV_DispatchThreadID) //256x256x32
{
	uint width = THREAD_Y_32Z, depth = 32, nwidth = 256;
	StdDefineNormal(id, decoder_conv4, decoder_conv4_statistic, width);
}

[numthreads(8,8,4)]
void DecoderInstance4(uint3 id: SV_DispatchThreadID) //id.xy=256 256x256x32
{
	uint width = 256, depth = 32;
	DefineDecoderInstRelu(id, 4)
}

[numthreads(8,8,4)]
void DecoderPad5(uint3 id: SV_DispatchThreadID) //id.xy=264 256x256x32->262x262x32
{
	uint pad = 3, width = 256, depth = 32;
	if (StdCheckRange(id, width + 2 * pad)) return;
	StdPad(id, width, depth, pad)
	decoder_conv5_pad[indx2] = decoder_conv4[indx];
}

float DotConv7x7(uint3 id, uint width, uint depth,uint z, int d_indx)
{
	float v = 0.0f;
	for(int i = 0; i < 7; i++)
	{
		[unroll(7)]
		for(int j = 0; j < 7; j++)
		{
			int c_indx = StdIndex(id.x + j, id.y + i, z, width, depth);
			v += decoder_conv5_pad[c_indx] * decoder_g_pred_c_Conv_weights[d_indx];
			d_indx++;
		}
	}
	return v;
}

[numthreads(8,8,1)]
void DecoderConv5(uint3 id: SV_DispatchThreadID) //id.xy=256
{
	//conv: （262x262x32) ->（256x256x3）
	uint width = 262, depth1 = 32, depth2 =3, ks = 7, stride = 1;
	float rgb[3];
	for(uint i = 0; i < depth2; i++)
	{
		float v= 0.0f;
		for(uint j = 0; j < depth1; j++)
		{
			int d_indx = (depth2 * j + i)* ks * ks;
			v += DotConv7x7(id, width, depth1, j, d_indx);
		}
		rgb[i] = sigmod(v);
	}
	uint2 coord = uint2(id.y, width - ks + stride -1 - id.x);
	decoder_destination[coord] = float4(rgb[0], rgb[1], rgb[2], 1);
}

/* StyleActive.compute */
#pragma kernel ActiveMain
#pragma kernel BufferMain

#include "libActive.cginc"

AppendStructuredBuffer<int> appendBuffer;
ConsumeStructuredBuffer<int> consumeBuffer;
RWTexture2D<float4> Destination;
float4 Color;

float relu_active(float x, float y, float eplison)
{
   return abs(relu(x) - y) < eplison ? 1 : 0;
}

float sigmod_active(float x, float y, float eplison)
{
   return abs(sigmod(x) - y) < eplison ? 1 : 0;
}

float tanh_active(float x, float y, float eplison)
{
   return abs(tanh(x) - y) < eplison ? 1 : 0;
}

[numthreads(8,8,1)]
void ActiveMain (uint3 id : SV_DispatchThreadID) 
{
   float r = relu_active(id.x - 128.0f, id.y - 128.0f, 1e-2);
   float g = sigmod_active((id.x - 128.0f)/32.0f, id.y/256.0f, 1e-2);
   float b = tanh_active((id.x - 128.0f)/32.0f, (id.y-128.0f)/128.0f, 1e-2);
   Destination[id.xy] = float4(r,g,b,1) * Color; 
}

[numthreads(8,1,1)]
void BufferMain (uint3 id : SV_DispatchThreadID) 
{
   if(id.x==2)
   {
      int value = consumeBuffer.Consume();
      appendBuffer.Append(value);
   }
   if(id.x>2)
   {
      appendBuffer.Append(id.x);
   }
}

_MainTex ("Texture", 2D) = "white" {}
Tags { "RenderType"="Opaque" }
LOD 100

#pragma vertex vert
#pragma fragment frag
#pragma multi_compile_fog

#include "UnityCG.cginc"

struct appdata
{
	float4 vertex : POSITION;
	float2 uv : TEXCOORD0;
};

struct v2f
{
	float2 uv : TEXCOORD0;
	UNITY_FOG_COORDS(1)
	float4 vertex : SV_POSITION;
};

sampler2D _MainTex;
float4 _MainTex_ST;

v2f vert (appdata v)
{
	v2f o;
	o.vertex = UnityObjectToClipPos(v.vertex);
	o.uv = TRANSFORM_TEX(v.uv, _MainTex);
	UNITY_TRANSFER_FOG(o,o.vertex);
	return o;
}

fixed4 frag (v2f i) : SV_Target
{
	fixed4 col = tex2D(_MainTex, i.uv);
	UNITY_APPLY_FOG(i.fogCoord, col);
	return col;
}
