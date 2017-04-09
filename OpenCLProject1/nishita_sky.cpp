#include"nishita_sky.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <limits> 
#include <string>

#ifndef M_PI
#define M_PI (3.14159265358979323846f)
#endif 

const float kInfinity = std::numeric_limits<float>::max();

const std::string tableRFile = "tableR.txt";
const std::string tableMFile = "tableM.txt";


size_t Atmosphere::rnum = 0;
size_t Atmosphere::cnum = 0;

const float Atmosphere::rstep = 10.f;
const float Atmosphere::cstep = .1f;

float* Atmosphere::tableM = NULL;
float* Atmosphere::tableR = NULL;
float* Atmosphere::itable = NULL;

const Vec3f Atmosphere::betaR(3.8e-6f, 13.5e-6f, 33.1e-6f);
const Vec3f Atmosphere::betaM(21e-6f);

bool solveQuadratic(float a, float b, float c, float& x1, float& x2)
{
	if (b == 0) {
		// Handle special case where the the two vector ray.dir and V are perpendicular
		// with V = ray.orig - sphere.centre
		if (a == 0) return false;
		x1 = 0; x2 = std::sqrtf(-c / a);
		return true;
	}
	float discr = b * b - 4 * a * c;

	if (discr < 0) return false;

	float q = (b < 0.f) ? -0.5f * (b - std::sqrtf(discr)) : -0.5f * (b + std::sqrtf(discr));
	x1 = q / a;
	x2 = c / q;

	return true;
}

// [comment]
// A simple routine to compute the intersection of a ray with a sphere
// [/comment]
bool raySphereIntersect(const Vec3f& orig, const Vec3f& dir, const float& radius, float& t0, float& t1)
{
	// They ray dir is normalized so A = 1 
	float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	float B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
	float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

	if (!solveQuadratic(A, B, C, t0, t1)) return false;

	if (t0 > t1) std::swap(t0, t1);

	return true;
}
const unsigned long dataStart = 10;
float* loadFromFile(const std::string filename, size_t& m, size_t& n) {
	std::ifstream inputFile(filename, std::ios::binary);
	if (inputFile.good()) {
		inputFile >> m >> n;
		float *table = new float[m*n];
		inputFile.seekg(dataStart);
		inputFile.read(reinterpret_cast<char *>(table), (unsigned long)m*n * sizeof(table[0]));
		inputFile.close();
		return table;
	}
	else {
		return NULL;
	}
}

void writeToFile(const std::string filename, float *table, size_t m, size_t n) {
	std::ofstream outputfile(filename, std::ios::binary);
	if (outputfile.is_open()) {
		outputfile << m << " " << n << std::endl;
		std::cout << outputfile.tellp() << std::endl;
		outputfile.write(reinterpret_cast<char *>(table), (unsigned long)m*n * sizeof(table[0]));
		outputfile.close();
	}
}
const float maxdegree = 180.f;
/*
tableR[100]	43058.7461	float
tableR[200]	22363.3086	float
tableR[250]	18149.5137	float
tableM[250]	812.942139	float
tableR[400]	11968.7158	float
tableM[400]	520.185120	float
*/
void Atmosphere::computeLightIntenseTable() {
	tableR = loadFromFile(tableRFile, rnum, cnum);
	tableM = loadFromFile(tableMFile, rnum, cnum);
	if (tableR == NULL || tableM == NULL) {
		rnum = (atmosphereRadius - earthRadius) / rstep,
			cnum = maxdegree / cstep;
		tableM = new float[rnum*cnum];
		tableR = new float[rnum*cnum];
		//look up table [R,Z]
		int i = 0, j = 0;
		for (float r = earthRadius; r < atmosphereRadius; r += rstep, i++) {
			j = 0;
			for (float c = 0; c < maxdegree; c += cstep, j++) {
				computeLightIntense(r, c, tableR[d2to1(i, j)], tableM[d2to1(i, j)]);
			}
		}
		writeToFile(tableRFile, tableR, rnum, cnum);
		writeToFile(tableMFile, tableM, rnum, cnum);
	}
}
//compute light intense for each altitude and C
void Atmosphere::computeLightIntense(float height, float c, float &lightR, float &lightM) const {
	static const Vec3f sunDirection = Vec3f(0.f, 1.f, 0.f);
	static const uint32_t numSamplesLight = 8;

	Vec3f samplePosition = Vec3f(0.f, height*cos(c / maxdegree*M_PI), height*sin(c / maxdegree*M_PI));
	float t0Light, t1Light;
	raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
	float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
	float opticalDepthLightR = 0.f, opticalDepthLightM = 0.f;
	uint32_t j;
	for (j = 0; j < numSamplesLight; ++j) {
		Vec3f samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
		float heightLight = samplePositionLight.length() - earthRadius;
		if (heightLight < 0) break;
		opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
		opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
		tCurrentLight += segmentLengthLight;
	}
	if (j == numSamplesLight) {
		lightR = opticalDepthLightR;
		lightM = opticalDepthLightM;
	}
	else {
		//std::cout << "hit at " << samplePosition << std::endl;
		lightM = 0.f;
		lightR = 0.f;
	}
}

float bilinearInterpolation(float ui, float vi, float *table, int r, int c) {
	ui = ui < 0 ? 0 : ui;
	vi = vi < 0 ? 0 : vi;
	ui = ui > r - 1 ? r - 1 : ui;
	vi = vi > c - 1 ? c - 1 : vi;
	float s = ui - (float)floor(ui), t = vi - (float)floor(vi);
	int ax = (int)floor(ui), dx = ax, ay = (int)floor(vi), by = ay;
	int cx = (int)ceil(ui), bx = cx, cy = (int)ceil(vi), dy = cy;
	if (table[cx*c + cy] <= 0.f || table[dx*c + dy] <= 0.f || table[bx*c + by] <= 0.f || table[ax*c + ay] <= 0.f) {
		return 0.f;
	}
	//std::cout << table[cx][cy] << " " << table[dx][dy] << " " << table[bx][by] << " " << table[ax][ay] << std::endl;
	float color = s*t*table[cx*c + cy]
		+ (1 - s)*t*table[dx*c + dy]
		+ s*(1 - t)*table[bx*c + by]
		+ (1 - t)*(1 - s)*table[ax*c + ay];
	return color;
}

// [comment]
// This is where all the magic happens. We first raymarch along the primary ray (from the camera origin
// to the point where the ray exits the atmosphere or intersect with the planetory body). For each
// sample along the primary ray, we then "cast" a light ray and raymarch along that ray as well.
// We basically shoot a ray in the direction of the sun.
// [/comment]
Vec3f Atmosphere::computeIncidentLight(const Vec3f& orig, const Vec3f& dir, float tmin, float tmax) const
{
	static uint32_t numSamples = 16;
	static uint32_t numSamplesLight = 8;
	static float g = 0.76f;

	float t0, t1;
	if (!raySphereIntersect(orig, dir, atmosphereRadius, t0, t1) || t1 < 0) return 0;
	if (t0 > tmin && t0 > 0) tmin = t0;
	if (t1 < tmax) tmax = t1;
	float segmentLength = (tmax - tmin) / numSamples;
	float tCurrent = tmin;
	Vec3f sumR(0), sumM(0); // mie and rayleigh contribution
	float opticalDepthR = 0, opticalDepthM = 0;
	float mu = dot(dir, sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
	float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
	float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
	for (uint32_t i = 0; i < numSamples; ++i) {
		Vec3f samplePosition = orig + (tCurrent + segmentLength * 0.5f) * dir;
		float height = samplePosition.length() - earthRadius;
		// compute optical depth for light
		float hr = exp(-height / Hr) * segmentLength;
		float hm = exp(-height / Hm) * segmentLength;
		opticalDepthR += hr;
		opticalDepthM += hm;
		// light optical depth
		float tCurrentLight = 0;
#if 1
		/////look up table
		////float lightRotation[3][3] =
		////{ { 1.f,	0.f,				0.f				},
		////{	0.f,	sunDirection[1],	sunDirection[2] },
		////{	0.f,	-sunDirection[2],	sunDirection[1] } };
		float lookupR, lookupM;
		float
			ny = sunDirection[1] * samplePosition[1] + sunDirection[2] * samplePosition[2],
			nz = -sunDirection[2] * samplePosition[1] + sunDirection[1] * samplePosition[2];
		float
			nl = std::sqrtf(nz * nz + ny * ny),
			nr = nl - earthRadius,
			nc = acos(ny / nl)*maxdegree / M_PI;
		lookupR = bilinearInterpolation(nr / rstep, nc / cstep, tableR, rnum, cnum);
		lookupM = bilinearInterpolation(nr / rstep, nc / cstep, tableM, rnum, cnum);
		//std::cout <<"Posistion:"<<samplePosition;
		//std::cout << "ny: " << ny << ", nz: " << nz << std::endl;
		//std::cout << "r: " << nr / rstep <<", c: "<< nc << std::endl;
		//std::cout << "lookup R:" << lookupR << std::endl;
		//std::cout << "lookup M:" << lookupM << std::endl;
		if (lookupR > 0.f) {
			Vec3f tau = betaR * (opticalDepthR + lookupR) + betaM * 1.1f * (opticalDepthM + lookupM);
			Vec3f attenuation(exp(-tau.x), exp(-tau.y), exp(-tau.z));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}
		else {
			//std::cout << "hit at " << samplePosition << " "<<std::endl;
		}
		/////-------------
#else
		float t0Light, t1Light;
		raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
		float segmentLengthLight = t1Light / numSamplesLight;
		float opticalDepthLightR = 0, opticalDepthLightM = 0;
		uint32_t j;
		for (j = 0; j < numSamplesLight; ++j) {
			Vec3f samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
			float heightLight = samplePositionLight.length() - earthRadius;
			if (heightLight < 0) {
				std::cout << "hit at:" << i << " " << j << std::endl;
				break;
			}
			opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
			tCurrentLight += segmentLengthLight;
		}
		std::cout << "cal R:" << opticalDepthLightR << std::endl;
		std::cout << "cal M:" << opticalDepthLightM << std::endl;
		if (j == numSamplesLight) {
			Vec3f tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
			Vec3f attenuation(exp(-tau.x), exp(-tau.y), exp(-tau.z));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}
#endif
		tCurrent += segmentLength;
	}

	// [comment]
	// We use a magic number here for the intensity of the sun (20). We will make it more
	// scientific in a future revision of this lesson/code
	// [/comment]
	return (sumR * betaR * phaseR + sumM * betaM * phaseM) * 20;
}

void renderSkydome(const Vec3f& sunDir, const char *filename)
{
	Atmosphere atmosphere(sunDir);
	auto t0 = std::chrono::high_resolution_clock::now();
#if 0
	// [comment]
	// Render fisheye
	// [/comment]
	const unsigned width = 512, height = 512;
	Vec3f *image = new Vec3f[width * height], *p = image;
	memset(image, 0x0, sizeof(Vec3f) * width * height);
	for (unsigned j = 0; j < height; ++j) {
		float y = 2.f * (j + 0.5f) / float(height - 1) - 1.f;
		for (unsigned i = 0; i < width; ++i, ++p) {
			float x = 2.f * (i + 0.5f) / float(width - 1) - 1.f;
			float z2 = x * x + y * y;
			if (z2 <= 1) {
				float phi = std::atan2(y, x);
				float theta = std::acos(1 - z2);
				Vec3f dir(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
				// 1 meter above sea level
				*p = atmosphere.computeIncidentLight(Vec3f(0, atmosphere.earthRadius + 1, 0), dir, 0, kInfinity);
			}
		}
		fprintf(stderr, "\b\b\b\b\%3d%c", (int)(100 * j / (width - 1)), '%');
	}
#else
	// [comment]
	// Render from a normal camera
	// [/comment]
	const unsigned width = 640, height = 480;
	Vec3f *image = new Vec3f[width * height], *p = image;
	memset(image, 0x0, sizeof(Vec3f) * width * height);
	float aspectRatio = width / float(height);
	float fov = 65;
	float angle = std::tan(fov * M_PI / 180 * 0.5f);
	unsigned numPixelSamples = 4;
	Vec3f orig(0, atmosphere.earthRadius + 1000, 30000); // camera position
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0, 1); // to generate random floats in the range [0:1]

	unsigned totalSample = numPixelSamples*numPixelSamples;
	float weight = 1.f / (float)totalSample;
	float *randomPattern = new float[totalSample];
	float *randomPos[2] = { new float[totalSample], new float[totalSample] };
	for (unsigned m = 0, si = 0; m < numPixelSamples; ++m) {
		for (unsigned n = 0; n < numPixelSamples; ++n) {
			randomPattern[si] = distribution(generator);
			randomPos[0][si] = m;
			randomPos[1][si] = n;
			si++;
		}
	}

	float widthf = 1.f / float(width), heightf = 1.f / float(height), numPixelSamplesf = 1.f / float(numPixelSamples);
	for (unsigned y = 0; y < height; ++y) {
		for (unsigned x = 0; x < width; ++x, ++p) {
			//16 point aa with uniform filter, need at least improve to Gaussian
			auto pixt0 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i<totalSample; i++) {
				float rayx = (2.f * (x + (randomPos[0][i] + randomPattern[i]) * numPixelSamplesf) * widthf - 1.f) * aspectRatio * angle;
				float rayy = (1.f - (y + (randomPos[1][i] + randomPattern[i]) * numPixelSamplesf) * heightf * 2.f) * angle;
				Vec3f dir(rayx, rayy, -1);
				normalize(dir);
				// [comment]
				// Does the ray intersect the planetory body? (the intersection test is against the Earth here
				// not against the atmosphere). If the ray intersects the Earth body and that the intersection
				// is ahead of us, then the ray intersects the planet in 2 points, t0 and t1. But we
				// only want to comupute the atmosphere between t=0 and t=t0 (where the ray hits
				// the Earth first). If the viewing ray doesn't hit the Earth, or course the ray
				// is then bounded to the range [0:INF]. In the method computeIncidentLight() we then
				// compute where this primary ray intersects the atmosphere and we limit the max t range 
				// of the ray to the point where it leaves the atmosphere.
				// [/comment]
				float t0, t1, tMax = kInfinity;
				if (raySphereIntersect(orig, dir, atmosphere.earthRadius, t0, t1) && t1 > 0)
					tMax = std::max(0.f, t0);
				// [comment]
				// The *viewing or camera ray* is bounded to the range [0:tMax]
				// [/comment]
				*p += atmosphere.computeIncidentLight(orig, dir, 0.f, tMax);
			}
			//std::cout << ((std::chrono::duration_cast<std::chrono::microseconds>)(std::chrono::high_resolution_clock::now() - pixt0)).count() << " us" << std::endl;
			*p *= weight;
		}
		fprintf(stderr, "\b\b\b\b%3d%c", (int)(100 * y / (width - 1)), '%');
	}
	free(randomPattern);
	free(randomPos[0]);
	free(randomPos[1]);
#endif

	std::cout << "\b\b\b\b" << ((std::chrono::duration<float>)(std::chrono::high_resolution_clock::now() - t0)).count() << " seconds" << std::endl;
	// Save result to a PPM image (keep these flags if you compile under Windows)
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	p = image;
	for (unsigned j = 0; j < height; ++j) {
		for (unsigned i = 0; i < width; ++i, ++p) {
#if 1
			// Apply tone mapping function
			(*p)[0] = (*p)[0] < 1.413f ? pow((*p)[0] * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-(*p)[0]);
			(*p)[1] = (*p)[1] < 1.413f ? pow((*p)[1] * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-(*p)[1]);
			(*p)[2] = (*p)[2] < 1.413f ? pow((*p)[2] * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-(*p)[2]);
#endif
			ofs << (unsigned char)(std::min(1.f, (*p)[0]) * 255)
				<< (unsigned char)(std::min(1.f, (*p)[1]) * 255)
				<< (unsigned char)(std::min(1.f, (*p)[2]) * 255);
		}
	}
	ofs.close();
	delete[] image;
}