//= ==================================================================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Varga Zsombor
// Neptun : AKCJOP
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

enum MaterialType {
	ROUGH,
	REFLECTIVE
};

const float epsilon = 0.0001f;
const int nSamples = 100;		// number of path samples per pixel
std::vector<vec3> samples;
float hiperRadius;

float rnd() { return (float)rand() / RAND_MAX; }

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) {
		type = t;
	}
};

vec3 SampleCalc() {
	float x, y, z;
	do {
		x = 2 * hiperRadius * rnd() - hiperRadius;
		y = 2 * hiperRadius * rnd() - hiperRadius;
	} while (powf(x, 2) + powf(y, 2) >= powf(hiperRadius, 2));  // reject if not in circle
	z = 0.95f;
	return vec3(x, y, z);
}

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI,
			kd = _kd,
			ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;

public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Ellipsoid : public Intersectable {
	vec3 center;
	float a, b, c;
	mat4 Q;
	float r1, r2, r3;

	Ellipsoid(vec3 _center, float _a, float _b, float _c, Material* _material) {
		center = _center;
		a = _a;
		b = _b;
		c = _c;
		r1 = _a;
		r2 = _b;
		r3 = _c;

		material = _material;

		Q = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, 1 / (c * c), 0,
			0, 0, 0, -1);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 S(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z, 1);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = gradf(vec4(hit.position.x - center.x, hit.position.y - center.y, hit.position.z - center.z, 1));
		hit.material = material;
		if (hit.position.z > 0.95) {
			hit.t = -1;
		}
		return hit;
	}

	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

};

struct Cylinder : public Intersectable {
	vec3 center;
	float a, b, radius, r1, r2;
	mat4 Q;

	Cylinder(vec3 _center, float _a, float _b, float _radius, Material* _material) {
		center = _center;
		a = _a;
		b = _b;
		r1 = a;
		r2 = b;
		radius = _radius;
		material = _material;

		Q = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, 0, 0,
			0, 0, 0, -10);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 S(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z, 1);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = gradf(vec4(hit.position.x - center.x, hit.position.y - center.y, hit.position.z - center.z, 1)) * (0.02f / radius);
		hit.material = material;
		if (hit.position.z > 0.3) {
			hit.t = -1;
		}
		return hit;
	}

	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}
};

struct Hyberboloid : public Intersectable {
	vec3 center;
	float a, b, c, r1, r2, r3;
	mat4 Q;

	Hyberboloid(vec3 _center, float _a, float _b, float _c, Material* _material) {
		center = _center;
		a = _a;
		b = _b;
		c = _c;
		r1 = _a;
		r2 = _b;
		r3 = _c;
		material = _material;

		Q = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, -1 / (c * c), 0,
			0, 0, 0, -1);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 S(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z, 1);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		if (t2 > 0 && (ray.start + ray.dir * t2).z < 5.0f && (ray.start + ray.dir * t2).z > 0.95f) {
			hit.t = t2;
		}
		else if ((ray.start + ray.dir * t1).z < 5.0f && (ray.start + ray.dir * t1).z > 0.95f) {
			hit.t = t1;
		}
		else { return hit; }
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(vec3(2 * (hit.position.x - center.x) / pow(r1, 2), 2 * (hit.position.y - center.y) / pow(r2, 2), -2 * (hit.position.z - center.z) / pow(r3, 2)));
		hit.material = material;
		return hit;

	}

};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = _direction;
		Le = _Le;
	}
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 SkyLa;

public:
	void build() {
		vec3 eye = vec3(0, 1.8, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		radiusCalc(vec3(2.0f, 2.0f, 1.0f), 0.95f);
		SkyLa = vec3(0.8f, 0.901f, 0.941f);

		vec3 lightDirection(0, 1, 0), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		for (int i = 0; i < nSamples; i++)
		{
			samples.push_back(SampleCalc());
		}

		//MATERIALS
		vec3 gn(0.17, 0.35, 1.5), gkappa(3.1, 2.7, 1.9);
		Material* gold = new ReflectiveMaterial(gn, gkappa);

		vec3 sn(0.14, 0.16, 0.13), skappa(4.1, 2.3, 3.1);
		Material* silver = new ReflectiveMaterial(sn, skappa);

		vec3 bd(0.1f, 0.1f, 0.6f), bs(2, 2, 2);
		Material* blue = new RoughMaterial(bd, bs, 50);

		vec3 wd(0.3f, 0.3f, 0.3f), ws(2, 2, 2);
		Material* wall = new RoughMaterial(wd, ws, 1);

		//Objectek
		objects.push_back(new Hyberboloid(vec3(0, 0, 0.95), hiperRadius, hiperRadius, hiperRadius, silver));
		objects.push_back(new Ellipsoid(vec3(0, 0, 0), 2.0f, 2.0f, 1.0f, wall));
		objects.push_back(new Cylinder(vec3(0.9, 0.2, 0), 0.1, 0.1, 1, blue));
		objects.push_back(new Cylinder(vec3(-1, -0.5, 0), 0.1, 0.1, 1, gold));
		objects.push_back(new Ellipsoid(vec3(0.5, -1.3, -1.7), 0.6f, 1.0f, 2.0f, silver));


	}

	void radiusCalc(vec3 radius, float m) {
		float z = m;
		hiperRadius = sqrtf((-1 * pow(z, 2) + 1) * pow(radius.x, 2));
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return vec3(0.3f, 0.3f, 0.3f);
		Hit hit = firstIntersect(ray);
		if (hit.t < 0 ) return SkyLa + lights.at(0)->Le * pow(dot(ray.dir, lights.at(0)->direction), 10);

		vec3 outRadiance(0, 0, 0);
		vec3 outDir;

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * vec3(0.3f, 0.3f, 0.3f);
			for (int i = 0; i < nSamples; i++) {

				outDir = samples.at(i) - (hit.position + hit.normal * epsilon);
				float cosDelta = dot(hit.normal, outDir);
				Ray shadowRay(hit.position + hit.normal * epsilon, outDir);

				if (cosDelta > 0 && !shadowIntersect(shadowRay)) {
					float A = hiperRadius * hiperRadius * M_PI;
					float dOmega = (A / nSamples) * (cosDelta / pow(length(outDir), 2));
					outRadiance = outRadiance + cosDelta * trace(Ray(hit.position + hit.normal * epsilon, outDir), depth + 1) * hit.material->kd * dOmega * hit.material->shininess;

				}
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;

		}

		return outRadiance;
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {

}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
