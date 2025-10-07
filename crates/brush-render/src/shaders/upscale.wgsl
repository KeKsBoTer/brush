#import helpers

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> rgba_img: array<u32>;
@group(0) @binding(2) var<storage, read> rgba_img_gradient: array<array<vec4<f32>,3>>;
@group(0) @binding(3) var<storage, read_write> out_img: array<u32>;


fn colormap_icefire(v:f32) -> vec3f{
  let vc = clamp(v, 0.0, 1.0);
  let blue  = vec3f(0.0, 0.0, 1.0);
  let black = vec3f(0.0, 0.0, 0.0);
  let red   = vec3f(1.0, 0.0, 0.0);
  let a = mix(blue, black, vc * 2.0);
  let b = mix(black, red, (vc - 0.5) * 2.0);
  return mix(a, b, step(0.5, vc));
}

fn spline_interp(z:mat2x2<f32>, dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>, p: vec2<f32>) -> f32
{
    let f = mat4x4<f32>(
        z[0][0], z[0][1], dy[0][0], dy[0][1],
        z[1][0], z[1][1], dy[1][0], dy[1][1],
        dx[0][0], dx[0][1], dxy[0][0], dxy[0][1],
        dx[1][0], dx[1][1], dxy[1][0], dxy[1][1]
    );
    let m = mat4x4<f32>(
        1., 0., 0., 0.,
        0., 0., 1., 0.,
        -3., 3., -2., -1.,
        2., -2., 1., 1.
    );
    let a = transpose(m) * f * (m);

    let tx = vec4<f32>(1., p.x, p.x * p.x, p.x * p.x * p.x);
    let ty = vec4<f32>(1., p.y, p.y * p.y, p.y * p.y * p.y);
    return dot(tx, a * ty);
}

fn loadImage(pos:vec2i)->vec4f{
    let location = vec2u(pos);
    let pix_id_source = location.x + location.y * uniforms.img_size.x;
    let color = rgba_img[pix_id_source];
    return unpack4x8unorm(color);
}


// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pix_loc = helpers::map_1d_to_2d(global_id.x, uniforms.tile_bounds.x);
    let pix_id = pix_loc.x + pix_loc.y * uniforms.target_size.x;

    let inside = pix_loc.x < uniforms.target_size.x && pix_loc.y < uniforms.target_size.y;

    if !inside {
        return;
    }

    let pixel_coord = vec2f(pix_loc);

    let uv = vec2f((f32(pix_loc.x) + 0.5) / f32(uniforms.target_size.x), (f32(pix_loc.y) + 0.5) / f32(uniforms.target_size.y));
    let in_width = f32(uniforms.img_size.x);
    let in_height = f32(uniforms.img_size.y);
    var uv_in = vec2f(fract(uv.x * in_width-0.5), fract(uv.y * in_height-0.5));
    if uv.x * in_width-0.5 < 0.0 {
        uv_in.x = fract(1.0-fract(uv.x * in_width-0.5));
    }
    if uv.y * in_height-0.5 < 0.0 {
        uv_in.y = fract(1.0-fract(uv.y * in_height-0.5));
    }
    

    let left_upper = vec2u(floor(uv * vec2f(in_width, in_height)-0.5));


    var z:  array<mat2x2<f32>,3>;
    var dx: array<mat2x2<f32>,3>;
    var dy: array<mat2x2<f32>,3>;
    var dxy:array<mat2x2<f32>,3>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = clamp(vec2i(left_upper) + vec2(i,j), vec2i(0), vec2i(uniforms.img_size) - vec2i(1));
            
            
            if uniforms.gradient_mode == 0u{
                let pix_id_source = u32(sample_pos.x) + u32(sample_pos.y) * uniforms.img_size.x;
                let z_v = loadImage(sample_pos).rgb;
                var dx_v = -rgba_img_gradient[pix_id_source][0].rgb;
                var dy_v = -rgba_img_gradient[pix_id_source][1].rgb;
                var dxy_v = rgba_img_gradient[pix_id_source][2].rgb;
                for (var c = 0u; c < 3u; c++) {
                    z[c][i][j] = z_v[c];
                    dx[c][i][j] = dx_v[c];
                    dy[c][i][j] = dy_v[c];
                    dxy[c][i][j] = dxy_v[c];
                }
            }else{
                let z_v = loadImage(sample_pos).rgb;
                let z_up = loadImage(clamp(sample_pos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(uniforms.img_size-1)));
                let z_left = loadImage(clamp(sample_pos + vec2<i32>(-1, 0), vec2<i32>(0), vec2<i32>(uniforms.img_size-1)));
                let z_right = loadImage(clamp(sample_pos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(uniforms.img_size-1)));
                let z_down = loadImage(clamp(sample_pos + vec2<i32>(0, -1), vec2<i32>(0), vec2<i32>(uniforms.img_size-1)));

                for (var c = 0u; c < 3u; c++) {
                    z[c][i][j] = z_v[c];
                    dx[c][i][j] = (z_right[c] - z_left[c]) * 0.5;
                    dy[c][i][j] = (z_up[c] - z_down[c]) * 0.5;
                    dxy[c][i][j] = (z_right[c] + z_left[c] - 2.0 * z_v[c]) * 0.5;
                }
            }
        }
    }

    var color_interp = vec3<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(uv_in.y, uv_in.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(uv_in.y, uv_in.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(uv_in.y, uv_in.x)),
    ); 

    var final_color:vec4f;
    switch uniforms.render_mode{
        case 1u{
            final_color = vec4f(colormap_icefire((dx[0][0][0]*5.+1.)/2.), 1.0);
        }
        case 2u{
            final_color = vec4f(colormap_icefire((dy[1][0][0]*5.+1.)/2.), 1.0);
        }
        case 3u{
            final_color = vec4f(colormap_icefire((dxy[2][0][0]*5.+1.)/2.), 1.0);
        }
        default{
            final_color = vec4f(color_interp, 1.0);
        }
    }

    out_img[pix_id] = pack4x8unorm(final_color);
}
