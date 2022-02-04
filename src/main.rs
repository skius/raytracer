use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::ops::{Add, Mul, Sub};
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};
use ::image::*;
use bvh::aabb::{AABB, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::BVH;
use bvh::{Point3, Vector3};
use bvh::ray::Ray;
use piston_window::*;
use rand::Rng;
use rayon::prelude::*;
use crate::types::Color;

struct RGBA([f64; 4]);

impl RGBA {
    fn new(r: f64, g: f64, b: f64, a: f64) -> RGBA {
        RGBA([r, g, b, a])
    }
}

impl Into<Rgba<u8>> for RGBA {
    fn into(self) -> Rgba<u8> {
        Rgba([
            (self.0[0] * 255.0).clamp(0.0, 255.0) as u8,
            (self.0[1] * 255.0).clamp(0.0, 255.0) as u8,
            (self.0[2] * 255.0).clamp(0.0, 255.0) as u8,
            (self.0[3] * 255.0).clamp(0.0, 255.0) as u8,
        ])
    }
}

#[derive(Debug, Copy, Clone)]
struct Vec2([f64; 2]);
#[derive(Debug, Copy, Clone)]
struct Vec3([f64; 3]);
#[derive(Debug, Copy, Clone)]
struct Vec3H([f64; 4]);
#[derive(Debug, Copy, Clone)]
struct Mat3H([[f64; 4]; 4]);

impl Vec2 {
    fn length(&self) -> f64 {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1]).sqrt()
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Vec3 {
    fn to_homogeneous(&self) -> Vec3H {
        Vec3H([self.0[0], self.0[1], self.0[2], 1.0])
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }

    fn length(&self) -> f64 {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]).sqrt()
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        let a = &self.0;
        let b = &other.0;
        Vec3([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }

    fn normalize(&self) -> Vec3 {
        let l = self.length();
        Vec3([self.0[0] / l, self.0[1] / l, self.0[2] / l])
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1], self.0[2] - rhs.0[2]])
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1], self.0[2] + rhs.0[2]])
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3([self * rhs.0[0], self * rhs.0[1], self * rhs.0[2]])
    }
}

impl Vec3H {
    fn homogeneous_divide(&self) -> Vec3 {
        Vec3([self.0[0] / self.0[3], self.0[1] / self.0[3], self.0[2] / self.0[3]])
    }
}

impl Mat3H {
    fn rot_y(theta: f64) -> Mat3H {
        let c = theta.cos();
        let s = theta.sin();
        Mat3H([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }

    fn rot_z(theta: f64) -> Mat3H {
        let c = theta.cos();
        let s = theta.sin();
        Mat3H([
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }

    fn rot_x(theta: f64) -> Mat3H {
        let c = theta.cos();
        let s = theta.sin();
        Mat3H([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c,   -s,  0.0],
            [0.0, s,   c,   0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }
}

impl Mul<Vec3H> for Mat3H {
    type Output = Vec3H;

    fn mul(self, other: Vec3H) -> Vec3H {
        Vec3H([
            self.0[0][0] * other.0[0] + self.0[0][1] * other.0[1] + self.0[0][2] * other.0[2] + self.0[0][3] * other.0[3],
            self.0[1][0] * other.0[0] + self.0[1][1] * other.0[1] + self.0[1][2] * other.0[2] + self.0[1][3] * other.0[3],
            self.0[2][0] * other.0[0] + self.0[2][1] * other.0[1] + self.0[2][2] * other.0[2] + self.0[2][3] * other.0[3],
            self.0[3][0] * other.0[0] + self.0[3][1] * other.0[1] + self.0[3][2] * other.0[2] + self.0[3][3] * other.0[3],
        ])
    }
}

impl Mul<Mat3H> for Mat3H {
    type Output = Mat3H;

    fn mul(self, rhs: Mat3H) -> Mat3H {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.0[i][k] * rhs.0[k][j];
                }
            }
        }
        Mat3H(result)
    }
}

const epsilon: f64 = 0.00000001;

fn ray_triangle_intersect(orig: Vec3, dir: Vec3, (v0, v1, v2): (Vec3, Vec3, Vec3)) -> Option<f64> {
    let edge0 = v1 - v0;
    let edge1 = v2 - v0;
    let n = edge0.cross(&edge1);
    let area = n.length();

    let ndotdir = n.dot(&dir);
    if ndotdir.abs() < epsilon {
        return None;
    }

    let d = -n.dot(&v0);

    let t = -(n.dot(&orig) + d) / ndotdir;
    if t < 0.0 {
        return None;
    }

    let p = orig + t * dir;
    let vp0 = p - v0;
    let c = edge0.cross(&vp0);
    if n.dot(&c) < 0.0 {
        return None;
    }

    let vp1 = p - v1;
    let edge1 = v2 - v1;
    let c = edge1.cross(&vp1);
    if n.dot(&c) < 0.0 {
        return None;
    }

    let vp2 = p - v2;
    let edge2 = v0 - v2;
    let c = edge2.cross(&vp2);
    if n.dot(&c) < 0.0 {
        return None;
    }

    Some(t)
}

fn intersects_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f64, f64, f64)> {
    let a_to_b = b - a;
    let a_to_c = c - a;

    // Begin calculating determinant - also used to calculate u parameter
    // u_vec lies in view plane
    // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
    let u_vec = dir.cross(&a_to_c);

    // If determinant is near zero, ray lies in plane of triangle
    // The determinant corresponds to the parallelepiped volume:
    // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
    let det = a_to_b.dot(&u_vec);

    // Only testing positive bound, thus enabling backface culling
    // If backface culling is not desired write:
    // det < EPSILON && det > -EPSILON
    if det < epsilon {
        return None;
    }

    let inv_det = 1.0 / det;

    // Vector from point a to ray origin
    let a_to_origin = origin - a;

    // Calculate u parameter
    let u = a_to_origin.dot(&u_vec) * inv_det;

    // Test bounds: u < 0 || u > 1 => outside of triangle
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    // Prepare to test v parameter
    let v_vec = a_to_origin.cross(&a_to_b);

    // Calculate v parameter and test bound
    let v = dir.dot(&v_vec) * inv_det;
    // The intersection lies outside of the triangle
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let dist = a_to_c.dot(&v_vec) * inv_det;

    if dist > epsilon {
        Some((dist, u, v))
    } else {
        None
    }
}

fn interp_sky(dir: Vec3) -> [f64; 3] {
    let r1 = 0.5;
    let r2 = 1.0;

    let g1 = 0.7;
    let g2 = 1.0;

    let b1 = 1.0;
    let b2 = 1.0;

    let t = dir.0[1];

    let mut res = (1.0 - t) * Vec3([r1, g1, b1]) + t * Vec3([r2, g2, b2]);

    // let light_dir = Vec3([1.0, 1.0, 1.0]).normalize();
    // res = (dir.dot(&light_dir) + 1.0)/2.0 * res;

    res.0
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Kind {
    Phong,
    Mirror,
}

#[derive(Clone)]
struct Object {
    polygon_mesh: PolygonMesh,
    kind: Kind,
    bvh: BVH,
    flat_triangles: Vec<FlatTriangle>,
    idx: usize,
}

impl Object {
    fn from_mesh(p: PolygonMesh, obj_idx: usize) -> Object {
        let pre = Instant::now();
        let mut flat_triangles = Vec::new();
        for (idx, &Triangle([vi1, vi2, vi3])) in p.triangles.iter().enumerate() {
            let v1 = p.vertices[vi1];
            let v2 = p.vertices[vi2];
            let v3 = p.vertices[vi3];

            let flat_tri = FlatTriangle {
                vertices: [v1, v2, v3],
                node_index: 0,
                triangle_idx: idx,
                obj_idx: obj_idx,
            };
            flat_triangles.push(flat_tri);
        }
        dbg!(Instant::now().duration_since(pre));
        let pre = Instant::now();
        let bvh = BVH::build(&mut flat_triangles);
        dbg!(Instant::now().duration_since(pre));
        Object {
            polygon_mesh: p,
            kind: Kind::Mirror,
            // kind: Kind::Phong,
            flat_triangles,
            bvh: bvh,
            idx: obj_idx,
        }
    }

    fn hit(&self, origin: Vec3, dir: Vec3) -> Option<Hit> {
        let ray = Ray::new(Point3::new(origin.0[0] as f32, origin.0[1] as f32, origin.0[2] as f32),
                           Vector3::new(dir.0[0] as f32, dir.0[1] as f32, dir.0[2] as f32));

        let hits = self.bvh.traverse(&ray, &self.flat_triangles);

        let least = hits.par_iter().filter_map(|flat_tri| {
            let v1 = flat_tri.vertices[0];
            let v2 = flat_tri.vertices[1];
            let v3 = flat_tri.vertices[2];

            // assert!(flat_tri.obj_idx == self.idx);

            // match ray_triangle_intersect(orig, dir, (v1, v2, v3)) {
            match intersects_triangle(origin, dir, v1, v2, v3) {
                Some(t) => Some((t, flat_tri.triangle_idx)),
                None => None,
            }
            // }).fold(((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });
        }).reduce(|| ((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });


        if least.0.0 == f64::INFINITY {
            None
        } else {
            Some(Hit::new(least.0.0, least.0.1, least.0.2, least.1))
        }

        // None
    }
}

#[derive(Debug, Copy, Clone)]
struct Hit {
    distance: f64,
    u: f64,
    v: f64,
    triangle_idx: usize,
}

impl Hit {
    fn new(distance: f64, u: f64, v: f64, triangle_idx: usize) -> Hit {
        Hit {
            distance,
            u,
            v,
            triangle_idx,
        }
    }
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

// fn ray_hit_mesh(origin: Vec3, dir: Vec3, mesh: &PolygonMesh) -> Option<Hit> {
//     let mut closest_hit = Hit::new(f64::INFINITY, 0.0, 0.0, 0);
//     for (triangle_idx, triangle) in mesh.triangles.iter().enumerate() {
//         if let Some((dist, u, v)) = intersects_triangle(origin, dir, triangle.0, triangle.1, triangle.2) {
//             if dist < closest_hit.distance {
//                 closest_hit = Hit::new(dist, u, v, triangle_idx);
//             }
//         }
//     }
//     if closest_hit.distance < f64::INFINITY {
//         Some(closest_hit)
//     } else {
//         None
//     }
// }

fn bary_interp<T>(a: T, b: T, c: T, u: f64, v: f64) -> T
where f64: Mul<T, Output = T>, T: Add<T, Output = T>
{
    (1.0 - u - v) * a + u * b + v * c
}

fn raytrace(origin: Vec3, dir: Vec3, objects: &[Object], depth: i32) -> [f64; 3] {
    if depth >= 2 {
        // println!("Depth {}", depth);
    }

    // Check if it hits an object
    let mut closest_hit = Hit::new(f64::INFINITY, 0.0, 0.0, 0);
    let mut closest_obj_idx = usize::MAX;
    for (obj_idx, obj) in objects.into_iter().enumerate(){
        let hit = obj.hit(origin, dir);
        if let Some(hit) = hit {
            if hit < closest_hit {
                closest_hit = hit;
                closest_obj_idx = obj_idx;
            }
        }
    }

    let light_pos = Vec3([5.0, 5.0, 0.0]);

    if closest_obj_idx == usize::MAX {
        return interp_sky(dir);
    } else {
        let obj = &objects[closest_obj_idx];
        let hit = closest_hit;
        let hit_pos = origin + hit.distance * dir;

        let Triangle([vi1, vi2, vi3]) = obj.polygon_mesh.triangles[hit.triangle_idx];
        let n1 = obj.polygon_mesh.vertice_normals[vi1];
        let n2 = obj.polygon_mesh.vertice_normals[vi2];
        let n3 = obj.polygon_mesh.vertice_normals[vi3];

        let normal = bary_interp(n1, n2, n3, hit.u, hit.v);
        // dbg!(normal);

        let camera_dir = (-1.0 * dir).normalize();
        let light_dir = (light_pos - hit_pos).normalize();

        let (u1, v1) = obj.polygon_mesh.uv_coords[obj.polygon_mesh.vertice_to_uv_idx[&vi1]];
        let (u2, v2) = obj.polygon_mesh.uv_coords[obj.polygon_mesh.vertice_to_uv_idx[&vi2]];
        let (u3, v3) = obj.polygon_mesh.uv_coords[obj.polygon_mesh.vertice_to_uv_idx[&vi3]];

        let text_u = bary_interp(u1, u2, u3, hit.u, hit.v);
        let text_v = bary_interp(v1, v2, v3, hit.u, hit.v);



        // assert!((0.0..1.0).contains(&text_u));
        // assert!((0.0..1.0).contains(&text_v));

        if obj.kind == Kind::Phong {

            // let light_dir = Vec3([1.0, -1.0, 1.0]).normalize();

            let diffuse = normal.dot(&light_dir);
            let diffuse = diffuse.max(0.0);

            let ambient = 0.1;

            let r = 2.0 * (light_dir.dot(&normal)) * normal - light_dir;
            let specular = r.dot(&camera_dir).max(0.0).powf(10.0);

            let text_width = obj.polygon_mesh.texture.width();
            let text_height = obj.polygon_mesh.texture.height();
            let text_x = (text_u * text_width as f64).floor() as u32;
            let text_y = text_height - (text_v * text_height as f64).floor() as u32;
            // dbg!((text_x, text_y));

            let Rgba([r, g, b, a]) = obj.polygon_mesh.texture.get_pixel(text_x, text_y);

            let r = r as f64 / 255.0;
            let g = g as f64 / 255.0;
            let b = b as f64 / 255.0;

            return [(diffuse + ambient + specular) * r, (diffuse + ambient + specular) * g, (diffuse + ambient + specular) * b];
        } else if obj.kind == Kind::Mirror {
            let r = 2.0 * (camera_dir.dot(&normal)) * normal - camera_dir;
            let orig = hit_pos;
            let dir = r.normalize();
            let [rr, rg, rb] = raytrace(orig, dir, objects, depth + 1);

            let specular = r.dot(&light_dir).max(0.0).powf(20.0);
            let specular = 0.0;

            let text_width = obj.polygon_mesh.texture.width();
            let text_height = obj.polygon_mesh.texture.height();
            let text_x = (text_u * text_width as f64).floor() as u32;
            let text_y = text_height - (text_v * text_height as f64).floor() as u32;
            // dbg!((text_x, text_y));

            if text_x >= text_width || text_x < 0 || text_y < 0 || text_y >= text_height {
                println!("{} {}", text_x, text_y);
            }

            let Rgba([r, g, b, a]) = obj.polygon_mesh.texture.get_pixel(text_x, text_y);
            let r = r as f64 / 255.0;
            let g = g as f64 / 255.0;
            let b = b as f64 / 255.0;

            // let [r, g, b] = [1.0; 3];

            return [rr * r + specular, rg * g + specular, rb * b + specular];
            // return [rr * r * 0.8 + r * 0.1 + specular, rg * g * 0.8 + g * 0.1 + specular, rb * b * 0.8 + b * 0.1 + specular];
            // return [rr * r * 0.8  + specular, rg * g * 0.8 + specular, rb * b * 0.8 + specular];
        }
    }

    [0.0; 3]
}

#[derive(Debug, Clone)]
struct PolygonMesh {
    vertices: Vec<Vec3>,
    vertice_normals: Vec<Vec3>,
    uv_coords: Vec<(f64, f64)>,
    vertice_to_uv_idx: HashMap<usize, usize>,
    triangles: Vec<Triangle>,
    texture: DynamicImage,
}

// impl PolygonMesh {
//     fn hit(&self, bvh: &BVH, origin: Vec3, dir: Vec3) -> Option<Hit> {
//         let ray = Ray::new(Point3::new(origin.0[0] as f32, origin.0[1] as f32, origin.0[2] as f32),
//                            Vector3::new(dir.0[0] as f32, dir.0[1] as f32, dir.0[2] as f32));
//
//         let hits = bvh.traverse(&ray);
//
//         let least = self.triangles.par_iter().enumerate().filter_map(|(i, &Triangle([vi1, vi2, vi3]))| {
//             let v1 = self.vertices[vi1];
//             let v2 = self.vertices[vi2];
//             let v3 = self.vertices[vi3];
//
//             // match ray_triangle_intersect(orig, dir, (v1, v2, v3)) {
//             match intersects_triangle(origin, dir, v1, v2, v3) {
//                 Some(t) => Some((t, i)),
//                 None => None,
//             }
//             // }).fold(((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });
//         }).reduce(|| ((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });
//
//
//         if least.0.0 == f64::INFINITY {
//             None
//         } else {
//             Some(Hit::new(least.0.0, least.0.1, least.0.2, least.1))
//         }
//     }
// }

#[derive(Debug, Clone)]
struct FlatTriangle {
    vertices: [Vec3; 3],
    node_index: usize,
    triangle_idx: usize,
    obj_idx: usize,
}

impl BHShape for FlatTriangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl Bounded for FlatTriangle {
    fn aabb(&self) -> AABB {
        let vertices = &self.vertices;
        let mut x_min = vertices[0].0[0];
        x_min = x_min.min(vertices[1].0[0]);
        x_min = x_min.min(vertices[2].0[0]);

        let mut x_max = vertices[0].0[0];
        x_max = x_max.max(vertices[1].0[0]);
        x_max = x_max.max(vertices[2].0[0]);

        let mut y_min = vertices[0].0[1];
        y_min = y_min.min(vertices[1].0[1]);
        y_min = y_min.min(vertices[2].0[1]);

        let mut y_max = vertices[0].0[1];
        y_max = y_max.max(vertices[1].0[1]);
        y_max = y_max.max(vertices[2].0[1]);

        let mut z_min = vertices[0].0[2];
        z_min = z_min.min(vertices[1].0[2]);
        z_min = z_min.min(vertices[2].0[2]);

        let mut z_max = vertices[0].0[2];
        z_max = z_max.max(vertices[1].0[2]);
        z_max = z_max.max(vertices[2].0[2]);

        AABB::with_bounds(Point3::new(x_min as f32, y_min as f32, z_min as f32), Point3::new(x_max as f32, y_max as f32, z_max as f32))
    }
}



impl PolygonMesh {
    fn rotate_y(&mut self, angle: f64) {
        let rot = Mat3H::rot_y(angle);

        for vertice in &mut self.vertices {
            *vertice = (rot * (*vertice).to_homogeneous()).homogeneous_divide();
        }
        for vertice_normal in &mut self.vertice_normals {
            *vertice_normal = (rot * (*vertice_normal).to_homogeneous()).homogeneous_divide();
        }
    }

    fn rotate_z(&mut self, angle: f64) {
        let rot = Mat3H::rot_z(angle);

        for vertice in &mut self.vertices {
            *vertice = (rot * (*vertice).to_homogeneous()).homogeneous_divide();
        }
        for vertice_normal in &mut self.vertice_normals {
            *vertice_normal = (rot * (*vertice_normal).to_homogeneous()).homogeneous_divide();
        }
    }

    fn rotate_x(&mut self, angle: f64) {
        let rot = Mat3H::rot_x(angle);

        for vertice in &mut self.vertices {
            *vertice = (rot * (*vertice).to_homogeneous()).homogeneous_divide();
        }
        for vertice_normal in &mut self.vertice_normals {
            *vertice_normal = (rot * (*vertice_normal).to_homogeneous()).homogeneous_divide();
        }
    }

    fn translate(&mut self, translation: Vec3) {
        for vertice in &mut self.vertices {
            *vertice = *vertice + translation;
        }
    }
}

fn render<const width: u32, const height: u32>(canvas: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, objects: &[Object]) {
    let fov_y = 60.0 * std::f64::consts::PI / 180.0;
    let aspect_ratio = width as f64 / height as f64;

    let mut coords = Vec::new();
    for u in 0..width {
        for v in 0..height {
            coords.push((u, v));
        }
    }

    coords.into_par_iter().map(|(u, v)| {
        let p_x = (2.0 * ((u as f64 + 0.5) / width as f64) - 1.0) * (fov_y / 2.0).tan() * aspect_ratio;
        let p_y = (1.0 - 2.0 * ((v as f64 + 0.5) / height as f64)) * (fov_y / 2.0).tan();

        let orig = Vec3([0.0, 0.0, 0.0]);
        let dir = Vec3([p_x, p_y, -1.0]).normalize();

        let [red, green, blue] = raytrace(orig, dir, objects, 0);



        let rgba = RGBA::new(red, green, blue, 1.0);
        ((u,v), rgba.into())
    }).collect::<Vec<_>>().into_iter().for_each(|((u, v), rgba)| {
        canvas.put_pixel(u, v, rgba);
    });
}

fn main() {
    const scale: u32 = 4;

    const width: u32 = scale * 160;
    const height: u32 = scale * 120;
    // const width: u32 = 9*160;
    // const height: u32 = 9*120;

    // let (width, height) = (5*160, 5*120);
    // let (width, height) = (20, 30);

    let mut start = Instant::now();

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("ray", (width, height))
            .exit_on_esc(true)
            .graphics_api(opengl)
            .build()
            .unwrap();

    let mut canvas = ImageBuffer::new(width, height);

    let spot_poly = load_obj("spot_triangulated.obj", "spot_texture.png");

    let mut texture_context = window.create_texture_context();

    let mut texture: G2dTexture = Texture::from_image(
        &mut texture_context,
        &canvas,
        &TextureSettings::new()
    ).unwrap();

    let mut t = 0.0;

    while let Some(e) = window.next() {
        println!("{:?}", e);
        if let Some(ue) = e.update_args() {
            t += ue.dt;
        }

        if let Some(re) = e.render_args() {

            // println!("{:?}", canvas.get_pixel(1,1));
            // canvas.put_pixel(1,1,RGBA::new(1.0, 0.0, 0.0, 1.0).into());
            // canvas.put_pixel(1,1,Rgba([255, 0, 0, 255]));
            //
            // for x in 0..width {
            //     for y in 0..height {
            //         canvas.put_pixel(x, y, Rgba([x as u8, y as u8, 0, 255]));
            //     }
            // }

            dbg!(Instant::now().duration_since(start));
            dbg!(re.ext_dt);
            // t += re.ext_dt;

            let mut objects = Vec::new();
            let mut phong = spot_poly.clone();
            // phong.rotate_z(80.0 * (t * 1250.0).sin() * std::f64::consts::PI / 180.0);
            // phong.rotate_x(t * 10000.0 * std::f64::consts::PI / 180.0);
            // phong.rotate_x(40.0 * (t * 2500.0).cos() * std::f64::consts::PI / 180.0);
            phong.rotate_y((150.0 + t*1000.0) * std::f64::consts::PI / 180.0);
            // phong.rotate_y((150.0) * std::f64::consts::PI / 180.0);
            phong.translate(Vec3([-1.0, 0.0, -2.5]));
            let obj = Object::from_mesh(phong, objects.len());
            objects.push(obj);

            // let pre = Instant::now();
            // let mut flat_triangles = Vec::new();
            // for (idx, &Triangle([vi1, vi2, vi3])) in phong.triangles.iter().enumerate() {
            //     let v1 = phong.vertices[vi1];
            //     let v2 = phong.vertices[vi2];
            //     let v3 = phong.vertices[vi3];
            //
            //     let flat_tri = FlatTriangle {
            //         vertices: [v1, v2, v3],
            //         node_index: 0,
            //         triangle_idx: idx,
            //         obj_idx: objects.len(),
            //     };
            //     flat_triangles.push(flat_tri);
            // }
            // dbg!(Instant::now().duration_since(pre));
            // let pre = Instant::now();
            // let bvh = BVH::build(&mut flat_triangles);
            // dbg!(Instant::now().duration_since(pre));
            // objects.push(Object {
            //     polygon_mesh: phong,
            //     kind: Kind::Mirror,
            //     // kind: Kind::Phong,
            //     flat_triangles,
            //     bvh: bvh,
            //     idx: objects.len(),
            // });

            let mut mirror = spot_poly.clone();
            mirror.rotate_y((220.0) * std::f64::consts::PI / 180.0);
            mirror.translate(Vec3([1.0, 0.0, -2.5]));
            let obj = Object::from_mesh(mirror, objects.len());
            objects.push(obj);

            // let pre = Instant::now();
            // let mut flat_triangles = Vec::new();
            // for (idx, &Triangle([vi1, vi2, vi3])) in mirror.triangles.iter().enumerate() {
            //     let v1 = mirror.vertices[vi1];
            //     let v2 = mirror.vertices[vi2];
            //     let v3 = mirror.vertices[vi3];
            //
            //     let flat_tri = FlatTriangle {
            //         vertices: [v1, v2, v3],
            //         node_index: 0,
            //         triangle_idx: idx,
            //         obj_idx: objects.len(),
            //     };
            //     flat_triangles.push(flat_tri);
            // }
            // dbg!(Instant::now().duration_since(pre));
            // let pre = Instant::now();
            // let bvh = BVH::build(&mut flat_triangles);
            // dbg!(Instant::now().duration_since(pre));
            // objects.push(Object {
            //     polygon_mesh: mirror,
            //     // kind: Kind::Phong,
            //     kind: Kind::Mirror,
            //     flat_triangles,
            //     bvh,
            //     idx: objects.len(),
            // });

            dbg!(Instant::now().duration_since(start));
            let pre = Instant::now();
            render::<width, height>(&mut canvas, &objects);
            dbg!(Instant::now().duration_since(pre));
            dbg!(Instant::now().duration_since(start));


            // render(width, height, &mut canvas);
            //
            texture.update(&mut texture_context, &canvas).unwrap();
            window.draw_2d(&e, |c, g, device| {
                // Update texture before rendering.
                texture_context.encoder.flush(device);
                clear([1.0; 4], g);
                image(&texture, c.transform, g);
            });
            //
            // sleep(Duration::from_secs(5));
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Triangle([usize; 3]);

fn load_obj(filename: impl AsRef<Path>, texturename: impl AsRef<Path>) -> PolygonMesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut uvs = Vec::new();
    let mut vertices_t = HashMap::new();

    let mut file = File::open(filename).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();

    for line in content.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();

        match parts[0] {
            "v" => {
                let v = parts[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>();
                vertices.push(Vec3([v[0], v[1], v[2]]));
            },
            "f" => {
                let f = parts[1..].iter().map(|x| x.split("/").map(|e| e.parse::<usize>().unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>();
                let vi1 = f[0][0] - 1;
                let vi2 = f[1][0] - 1;
                let vi3 = f[2][0] - 1;

                let uv1 = f[0][1] - 1;
                let uv2 = f[1][1] - 1;
                let uv3 = f[2][1] - 1;

                triangles.push(Triangle([vi1, vi2, vi3]));

                if vertices_t.contains_key(&vi1) {
                    println!("ALREADY HAS: {}, NEW: {}", vertices_t[&vi1], uv1);
                }

                vertices_t.insert(vi1, uv1);
                vertices_t.insert(vi2, uv2);
                vertices_t.insert(vi3, uv3);
            },
            "vt" => {
                let v = parts[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>();
                uvs.push((v[0], v[1]));
            },
            _ => (),
        }
    }

    let mut verts_n = vec![Vec::new(); vertices.len()];

    for &Triangle([idx1, idx2, idx3]) in &triangles {
        let v1 = vertices[idx1];
        let v2 = vertices[idx2];
        let v3 = vertices[idx3];

        let normal = (v2 - v1).cross(&(v3 - v1)).normalize();
        verts_n[idx1].push(normal);
        verts_n[idx2].push(normal);
        verts_n[idx3].push(normal);
    }

    let mut verts_normals = Vec::new();
    for v in verts_n {
        let mut normal = Vec3([0.0, 0.0, 0.0]);
        for n in v {
            normal = normal + n;
        }
        normal = normal.normalize();
        verts_normals.push(normal);
    }

    PolygonMesh {
        vertices,
        vertice_normals: verts_normals,
        uv_coords: uvs,
        triangles,
        vertice_to_uv_idx: vertices_t,
        texture: ::image::io::Reader::open(texturename).unwrap().decode().unwrap(),
    }
}
