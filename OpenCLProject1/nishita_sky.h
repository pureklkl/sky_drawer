#pragma once

#include <iostream>

template<typename T>
class Vec3
{
public:
	Vec3() : x(0), y(0), z(0) {}
	Vec3(T xx) : x(xx), y(xx), z(xx) {}
	Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	Vec3 operator * (const T& r) const { return Vec3(x * r, y * r, z * r); }
	Vec3 operator * (const Vec3<T> &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
	Vec3 operator + (const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
	Vec3 operator - (const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
	template<typename U>
	Vec3 operator / (const Vec3<U>& v) const { return Vec3(x / v.x, y / v.y, z / v.z); }
	friend Vec3 operator / (const T r, const Vec3& v)
	{
		return Vec3(r / v.x, r / v.y, r / v.z);
	}
	const T& operator [] (size_t i) const { return (&x)[i]; }
	T& operator [] (size_t i) { return (&x)[i]; }
	T length2() const { return x * x + y * y + z * z; }
	T length() const { return std::sqrt(length2()); }
	Vec3& operator += (const Vec3<T>& v) { x += v.x, y += v.y, z += v.z; return *this; }
	Vec3& operator *= (const float& r) { x *= r, y *= r, z *= r; return *this; }
	friend Vec3 operator * (const float&r, const Vec3& v)
	{
		return Vec3(v.x * r, v.y * r, v.z * r);
	}
	friend std::ostream& operator << (std::ostream& os, const Vec3<T>& v)
	{
		os << v.x << " " << v.y << " " << v.z << std::endl; return os;
	}
	T x, y, z;
};

template<typename T>
void normalize(Vec3<T>& vec)
{
	T len2 = vec.length2();
	if (len2 > 0) {
		T invLen = 1 / std::sqrt(len2);
		vec.x *= invLen, vec.y *= invLen, vec.z *= invLen;
	}
}

template<typename T>
T dot(const Vec3<T>& va, const Vec3<T>& vb)
{
	return va.x * vb.x + va.y * vb.y + va.z * vb.z;
}

using Vec3f = Vec3<float>;

// [comment]
// The atmosphere class. Stores data about the planetory body (its radius), the atmosphere itself
// (thickness) and things such as the Mie and Rayleigh coefficients, the sun direction, etc.
// [/comment]
class Atmosphere
{
public:
	Atmosphere(
		Vec3f sd = Vec3f(0, 1, 0),
		float er = 6360e3, float ar = 6420e3,
		float hr = 7994, float hm = 1200) :
		sunDirection(sd),
		earthRadius(er),
		atmosphereRadius(ar),
		Hr(hr),
		Hm(hm)
	{
		if (tableR == NULL || tableM == NULL) {
			computeLightIntenseTable();
		}
	}

	Vec3f computeIncidentLight(const Vec3f& orig, const Vec3f& dir, float tmin, float tmax) const;
	void computeLightIntenseTable();
	void computeLightIntense(float height, float c, float &lightR, float &lightM) const;
	int d2to1(int i, int j) {
		return i * cnum + j;
	}
	static void clearTable() {
		if (tableR != NULL) {
			delete[] tableR;
		}
		if (tableM != NULL) {
			delete[] tableM;
		}
		if (itable != NULL) {
			delete[] itable;
		}
	}
	
	size_t getRnum() {
		return rnum;
	}

	size_t getCnum() {
		return cnum;
	}

	float* getIntegratedTable() {
		if (itable == NULL) {
			size_t total = rnum*cnum * 2;
			itable = new float[total];
			for (size_t i = 0, j = 0; i < total; j++) {
				itable[i] = tableR[j];
				i++;
				itable[i] = tableM[j];
				i++;
			}
		}
		return itable;
	}

	Vec3f sunDirection;     // The sun direction (normalized)
	float earthRadius;      // In the paper this is usually Rg or Re (radius ground, eart)
	float atmosphereRadius; // In the paper this is usually R or Ra (radius atmosphere)
	float Hr;               // Thickness of the atmosphere if density was uniform (Hr)
	float Hm;               // Same as above but for Mie scattering (Hm)

private:

	static size_t rnum;
	static size_t cnum;

	static float *tableR;
	static float *tableM;
	static float *itable;

	static const Vec3f betaR;
	static const Vec3f betaM;

	static const float rstep;
	static const float cstep;
};
