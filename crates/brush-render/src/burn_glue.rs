use burn::tensor::{DType, ops::FloatTensor};
use burn_cubecl::{BoolElement, fusion::FusionCubeRuntime};
use burn_fusion::{
    Fusion, FusionHandle,
    client::FusionClient,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr};
use burn_wgpu::WgpuRuntime;
use glam::Vec3;

use crate::{
    MainBackendBase, SplatForward,
    camera::Camera,
    render::{calc_tile_bounds, max_intersections, render_forward},
    render_aux::RenderAux,
    shaders,
};

// Implement forward functions for the inner wgpu backend.
impl SplatForward<Self> for MainBackendBase {
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        opacity: FloatTensor<Self>,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        render_forward(
            camera, img_size, means, log_scales, quats, sh_coeffs, opacity, background, bwd_info,
        )
    }
}

impl SplatForward<Self> for Fusion<MainBackendBase> {
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        opacity: FloatTensor<Self>,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        #[derive(Debug)]
        struct CustomOp {
            cam: Camera,
            img_size: glam::UVec2,
            bwd_info: bool,
            background: Vec3,
            desc: CustomOpIr,
        }

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [means, log_scales, quats, sh_coeffs, opacity] = inputs;
                let [
                    projected_splats,
                    uniforms_buffer,
                    num_intersections,
                    tile_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                    out_img,
                    out_img_gradient,
                    visible,
                ] = outputs;

                let (img, aux) = MainBackendBase::render_splats(
                    &self.cam,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(means),
                    h.get_float_tensor::<MainBackendBase>(log_scales),
                    h.get_float_tensor::<MainBackendBase>(quats),
                    h.get_float_tensor::<MainBackendBase>(sh_coeffs),
                    h.get_float_tensor::<MainBackendBase>(opacity),
                    self.background,
                    self.bwd_info,
                );

                // Register output.
                h.register_float_tensor::<MainBackendBase>(&out_img.id, img);
                h.register_float_tensor::<MainBackendBase>(&out_img_gradient.id, aux.img_gradient);
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    aux.projected_splats,
                );
                h.register_int_tensor::<MainBackendBase>(&uniforms_buffer.id, aux.uniforms_buffer);
                h.register_int_tensor::<MainBackendBase>(
                    &num_intersections.id,
                    aux.num_intersections,
                );
                h.register_int_tensor::<MainBackendBase>(&tile_offsets.id, aux.tile_offsets);
                h.register_int_tensor::<MainBackendBase>(
                    &compact_gid_from_isect.id,
                    aux.compact_gid_from_isect,
                );
                h.register_int_tensor::<MainBackendBase>(
                    &global_from_compact_gid.id,
                    aux.global_from_compact_gid,
                );

                h.register_float_tensor::<MainBackendBase>(&visible.id, aux.visible);
            }
        }

        let client = means.client.clone();

        let num_points = means.shape[0];

        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / 4;
        let uniforms_size = size_of::<shaders::helpers::RenderUniforms>() / 4;
        let tile_bounds = calc_tile_bounds(img_size);
        let max_intersects = max_intersections(img_size, num_points as u32);

        // If render_u32_buffer is true, we render a packed buffer of u32 values, otherwise
        // render RGBA f32 values.
        let channels = if bwd_info { 4 } else { 1 };

        let out_img = client.tensor_uninitialized(
            vec![img_size.y as usize, img_size.x as usize, channels],
            if bwd_info { DType::F32 } else { DType::U32 },
        );
        let out_img_gradient = client.tensor_uninitialized(
            vec![img_size.y as usize, img_size.x as usize, 4*3],
            DType::F32,
        ); 

        let visible_shape = if bwd_info { vec![num_points] } else { vec![1] };

        let aux = RenderAux::<Self> {
            projected_splats: client.tensor_uninitialized(vec![num_points, proj_size], DType::F32),
            uniforms_buffer: client.tensor_uninitialized(vec![uniforms_size], DType::U32),
            num_intersections: client.tensor_uninitialized(vec![1], DType::U32),
            tile_offsets: client.tensor_uninitialized(
                vec![tile_bounds.y as usize, tile_bounds.x as usize, 2],
                DType::U32,
            ),
            compact_gid_from_isect: client
                .tensor_uninitialized(vec![max_intersects as usize], DType::U32),
            global_from_compact_gid: client.tensor_uninitialized(vec![num_points], DType::U32),
            visible: client.tensor_uninitialized(visible_shape, DType::F32),
            img_size,
            img_gradient: out_img_gradient,
        };

        let mut stream = OperationStreams::default();
        let input_tensors = [means, log_scales, quats, sh_coeffs, opacity];
        let output_tensors = [
            &aux.projected_splats,
            &aux.uniforms_buffer,
            &aux.num_intersections,
            &aux.tile_offsets,
            &aux.compact_gid_from_isect,
            &aux.global_from_compact_gid,
            &out_img,
            &aux.visible,
            &aux.img_gradient,
        ];
        for inp in &input_tensors {
            stream.tensor(inp);
        }
        let desc = CustomOpIr::new(
            "render_splats",
            &input_tensors.map(|t| t.into_ir()),
            &output_tensors.map(|t| t.to_ir_out()),
        );
        let op = CustomOp {
            cam: cam.clone(),
            img_size,
            bwd_info,
            background,
            desc: desc.clone(),
        };
        client.register(stream, OperationIr::Custom(desc), op);
        (out_img, aux)
    }
}
