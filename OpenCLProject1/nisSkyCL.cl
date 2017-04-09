

// TODO: Add OpenCL kernel code here.
#ifndef M_PI
#define M_PI (3.14159265358979323846f)
#endif 

#define EARTH_RADIUS (6360e3)
#define AR_RADIUS (6420e3)
#define HR (7994)
#define HM (1200)

#define MAXDEGREE 180.f

#define numSamples 16
#define numSamplesLight 8
#define MSCARTTING_G 0.76f

#define RSTEP 10.0f
#define CSTEP 0.1f

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

//camera position
__constant float3 orig = {0.f, EARTH_RADIUS + 1000.f, 30000.f };

//scartting parameters
__constant float3 betaR = {3.8e-6f, 13.5e-6f, 33.1e-6f };
__constant float3 betaM = {21e-6f, 21e-6f, 21e-6f };

float solveQuadratic(float a, float b, float c, float *x1, float *x2)
{
	if (b == 0) {
		// Handle special case where the the two vector ray.dir and V are perpendicular
		// with V = ray.orig - sphere.centre
		if (a == 0) return -1.f;
		*x1 = 0; *x2 = sqrt(-c / a);
		return 1.f;
	}
	float discr = b * b - 4 * a * c;

    float discr_s = select(1.f, -1.f, discr<0.f);
    float discr_abs = discr_s * discr;

	float q = (b < 0.f) ? -0.5f * (b - sqrt(discr_abs)) : -0.5f * (b + sqrt(discr_abs));
	*x1 = q / a;
	*x2 = c / q;

	return discr_s;
}

void swap(float *t0, float *t1){ 
	float tmp = *t0;
	*t0 = *t1;
	*t1 = tmp;
}

bool raySphereIntersect(const float3 orig, const float3 dir, const float radius, float* t0, float* t1)
{
	// They ray dir is normalized so A = 1 
	float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	float B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
	float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

	float valid = solveQuadratic(A, B, C, t0, t1);

	float maxt = max(*t0, *t1), mint = min(*t0, *t1);
	*t0 = mint;
	*t1 = maxt;

	return valid >= 0.f;
}

int valid(float ui, float vi, float r, float c, __read_only image2d_t lightTable){ 
	ui = clamp(ui, 0.f, r - 1.f);
	vi = clamp(vi, 0.f, c - 1.f);
	float ax = floor(ui), dx = ax, ay = floor(vi), by = ay;
	float cx = ceil(ui), bx = cx, cy = ceil(vi), dy = cy;
	float4 
		pa = read_imagef(lightTable, sampler, (float2)(ay, ax)),
		pb = read_imagef(lightTable, sampler, (float2)(by, bx)),
		pc = read_imagef(lightTable, sampler, (float2)(cy, cx)),
		pd = read_imagef(lightTable, sampler, (float2)(dy, dx));
	return (int)pa.x<=0.f|(int)pb.x<=0.f|(int)pc.x<=0.f|(int)pc.x<=0.f|(int)pd.x<=0.f;	
}

float3 computeIncidentLight(const float3 dir, const float3 sunDirection, float tMin, float tMax, __read_only image2d_t lightTable){ 
	float t0, t1;
	
	bool inter = (int)(!raySphereIntersect(orig, dir, AR_RADIUS, &t0, &t1)) | (int)(t1 < 0);
	tMin = max(max(t0, 0.f), tMin); 
	tMax = min(t1, tMax);
	
	float segmentLength = (tMax - tMin) / numSamples;
	float tCurrent = tMin;

	float3 sumR = {0.f, 0.f, 0.f }, sumM = {0.f, 0.f, 0.f };
	float opticalDepthR = 0, opticalDepthM = 0;
	float mu = dot(dir, sunDirection);
	
	float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
	float phaseM = 3.f / (8.f * M_PI) * ((1.f - MSCARTTING_G * MSCARTTING_G) * (1.f + mu * mu)) / ((2.f + MSCARTTING_G * MSCARTTING_G) * pow(1.f + MSCARTTING_G * MSCARTTING_G - 2.f * MSCARTTING_G * mu, 1.5f));
	
	for(size_t i = 0; i < numSamples; i++){ 
		float3 samplePosition = orig + dir * (tCurrent + segmentLength * 0.5f);
		float height = length(samplePosition) - (float)EARTH_RADIUS;
		// compute optical depth for light
		float hr = exp(-height / HR) * segmentLength;
		float hm = exp(-height / HM) * segmentLength;
		opticalDepthR += hr;
		opticalDepthM += hm;
		// light optical depth
		float tCurrentLight = 0;

		float
			ny = sunDirection.y * samplePosition.y + sunDirection.z * samplePosition.z,
			nz = -sunDirection.z * samplePosition.y + sunDirection.y * samplePosition.z;
		float 
			nl = sqrt(nz*nz+ny*ny),
			nr = (nl - (float)EARTH_RADIUS) / RSTEP,
			nc = acos(ny / nl)*MAXDEGREE / M_PI / CSTEP;
		int w = get_image_width(lightTable),
		    h = get_image_height(lightTable);

		float4 RM = read_imagef(lightTable, sampler, (float2)(nc, nr));	
		

		int rValid = valid(nr, nc, h, w, lightTable);
		rValid = select(0, -1, rValid==1);
		float3 tau = betaR * (opticalDepthR + RM.x) + betaM * 1.1f * (opticalDepthM + RM.y);
		float3 attenuation = {exp(-tau.x), exp(-tau.y), exp(-tau.z) };
		sumR += select(attenuation * hr, (float3)(0.f, 0.f, 0.f), (int3)(rValid, rValid, rValid));
		sumM += select(attenuation * hm, (float3)(0.f, 0.f, 0.f), (int3)(rValid, rValid, rValid));

		tCurrent += segmentLength;
	}

	return (sumR * betaR * phaseR + sumM * betaM * phaseM) * 20.f;
}

/*
		params[0] = (void *)&lightTable;
		params[1] = (void *)&sampleDataWH;
		params[2] = (void *)&sampleData;
		params[3] = (void *)&sunDir;
		params[4] = (void *)&superSample;
		outputHW
		angle
		sunDir
*/
__kernel void nisSkyCL(__read_only image2d_t lightTable, 
					 __constant size_t *sampleWH, 
					 __constant float *sampleData,
					 __write_only image2d_t superSample,
					 float2 outputWH,
					 float angle, 
					 float3 sunDir
					 ) {

	 const int2 pos = {get_global_id(0), get_global_id(1)};
	 float3 ray = {floor((float)pos.x/sampleWH[0]), floor((float)pos.y/sampleWH[1]), -1.f };
	 uint sx = (pos.x % sampleWH[0]) * 3, sy = pos.y % sampleWH[1];
	 float3 rand = {sampleData[sy*sampleWH[0]*3+sx], sampleData[sy*sampleWH[0]*3+sx+1], 0.f};
	 float weight = sampleData[sy*sampleWH[0]*3+sx+2];
	 
	 float3 trans = {2.f/outputWH.x, -2.f/outputWH.y, 1.f };
	 float aspectRatio = outputWH.x/(float)outputWH.y;
	 ray = ((ray+rand)*trans+(float3)(-1.f, 1.f, 0.f))*(float3)(aspectRatio*angle, angle, 1.f);
	 ray = normalize(ray);//fast_normailze?		 
	 
	 float t0, t1, tMax = FLT_MAX;
	 bool inter = (int)raySphereIntersect(orig, ray, EARTH_RADIUS, &t0, &t1) & (int)(t1>0) ;
	 tMax = select(tMax, max(t0, 0.f), inter);

	 float4 p = { computeIncidentLight(ray, sunDir, 0.f, tMax, lightTable) * weight, 1.f};

	 write_imagef(superSample, pos, p);
}