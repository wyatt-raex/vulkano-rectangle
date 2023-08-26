use std::sync::Arc;

use vulkano::{
  buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
  command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    RenderingAttachmentInfo, RenderingInfo,
  },
  device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
    QueueCreateInfo, QueueFlags,
  },
  image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
  instance::{Instance, InstanceCreateInfo},
  memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
  pipeline::{
    graphics::{
      input_assembly::{InputAssemblyState, PrimitiveTopology},
      render_pass::PipelineRenderingCreateInfo,
      vertex_input::Vertex,
      viewport::{Viewport, ViewportState},
    },
    GraphicsPipeline,
  },
  render_pass::{LoadOp, StoreOp},
  swapchain::{
    acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
  },
  sync::{self, FlushError, GpuFuture},
  Version, VulkanLibrary, shader::ShaderModule,
};

use vulkano_win::VkSurfaceBuild;

use winit::{
  event::{Event, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
  window::{Window, WindowBuilder},
};

const APP_VERSION: Version = Version {
  major: 0,
  minor: 0,
  patch: 1,
};


fn main() {
  let library = VulkanLibrary::new().unwrap();
  let required_extensions = vulkano_win::required_extensions(&library);

  // Create the instance
  let instance = Instance::new(
    library,
    InstanceCreateInfo{
      application_name: Some(String::from("vulkano-rect")),
      application_version: APP_VERSION, 
      enabled_extensions: required_extensions,
      enumerate_portability: true,
      ..Default::default()
    },
  ).unwrap();

  // Create the surface
  let event_loop = EventLoop::new();
  let surface = WindowBuilder::new()
    .build_vk_surface(&event_loop, instance.clone())
    .unwrap();

  // Select the device and queues to use
  let (device, mut queues) = create_device(&instance, &surface);
  let queue = queues.next().unwrap();

  let (mut swapchain, images) = create_swapchain(&device, &surface);

  // Create our Vertex buffer
  let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

  // TODO: Figure out a way to declare this outside main
  #[derive(BufferContents, Vertex)]
  #[repr(C)]
  struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
  }

  let vertices = [
    Vertex {
      position: [-0.5, 0.5],
    },
    Vertex {
      position: [-0.5, -0.5],
    },
    Vertex {
      position: [0.5, -0.5],
    },
    Vertex {
      position: [0.5, 0.5],
    },
    
  ];
  let vertex_buffer = Buffer::from_iter(
    &memory_allocator,
    BufferCreateInfo {
      usage: BufferUsage::VERTEX_BUFFER,
      ..Default::default()
    },
    AllocationCreateInfo {
      usage: MemoryUsage::Upload,
      ..Default::default()
    },
    vertices,
  )
  .unwrap();

  // Create the shaders 
  mod vs {
    vulkano_shaders::shader! {
      ty: "vertex",
      src: r"
        #version 450

        layout(location = 0) in vec2 position;

        void main() {
          gl_Position = vec4(position, 0.0, 1.0);
        }
      ",
    }
  }

  mod fs {
    vulkano_shaders::shader! {
      ty: "fragment",
      src: r"
        #version 450

        layout(location = 0) out vec4 f_color;

        void main() {
          f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
      ",
    }
  }

  let vs = vs::load(device.clone()).unwrap();
  let fs = fs::load(device.clone()).unwrap();

  // Graphics Pipeline creation
  let pipeline = create_graphics_pipeline(&device, &swapchain, &vs, &fs);

  let mut viewport = Viewport {
    origin: [0.0, 0.0],
    dimensions: [0.0, 0.0],
    depth_range: 0.0..1.0,
  };

  let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);

  let command_buffer_allocator =
    StandardCommandBufferAllocator::new(device.clone(), Default::default());

  let mut recreate_swapchain = false;
  let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

  event_loop.run(move |event, _, control_flow| {
    match event {
      Event::WindowEvent {
        event: WindowEvent::CloseRequested,
        ..
      } => {
        *control_flow = ControlFlow::Exit;
      }

      Event::WindowEvent {
        event: WindowEvent::Resized(_),
        ..
      } => {
        recreate_swapchain = true;
      }

      Event::RedrawEventsCleared => {
        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if recreate_swapchain {
          let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

          let (new_swapchain, new_images) = 
            match swapchain.recreate(SwapchainCreateInfo{
              image_extent: window.inner_size().into(),
              ..swapchain.create_info()
            }) {
              Ok(r) => r,
              Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
              Err(e) => panic!("failed to recreate swapchain: {e}"),
            };

          swapchain = new_swapchain;

          attachment_image_views = window_size_dependent_setup(&new_images, &mut viewport);
          recreate_swapchain = false;
        }

        draw();

      }

      _ => (),
    }
  });
}


fn draw() {
  let (image_index, suboptimal, acquire_future) =
    match acquire_next_image(swapchain.clone(), None) {
      Ok(r) => r,
      Err(AcquireError::OutOfDate) => {
        recreate_swapchain = true;
        return;
      }
      Err(e) => panic!("failed to acquire next image: {e}"),
    };

  if suboptimal { recreate_swapchain = true; }

  let mut builder = AutoCommandBufferBuilder::primary(
    &command_buffer_allocator,
    queue.queue_family_index(),
    CommandBufferUsage::OneTimeSubmit,
  )
  .unwrap();

  builder
    .begin_rendering(RenderingInfo {
      color_attachments: vec![Some(RenderingAttachmentInfo {
        load_op: LoadOp::Clear,
        store_op: StoreOp::Store,
        clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
        ..RenderingAttachmentInfo::image_view(
          attachment_image_views[image_index as usize].clone(),
        )
      })],

      ..Default::default()
    })
    .unwrap()
    .set_viewport(0, [viewport.clone()])
    .bind_pipeline_graphics(pipeline.clone())
    .bind_vertex_buffers(0, vertex_buffer.clone())
    .draw(vertex_buffer.len() as u32, 1, 0, 0)
    .unwrap()
    .end_rendering()
    .unwrap();

  let command_buffer = builder.build().unwrap();

  let future = previous_frame_end
    .take()
    .unwrap()
    .join(acquire_future)
    .then_execute(queue.clone(), command_buffer)
    .unwrap()
    .then_swapchain_present(
      queue.clone(),
      SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
    )
    .then_signal_fence_and_flush();

  match future {
    Ok(future) => {
      previous_frame_end = Some(future.boxed());
    }
    Err(FlushError::OutOfDate) => {
      recreate_swapchain = true;
      previous_frame_end = Some(sync::now(device.clone()).boxed());
    }
    Err(e) => {
      println!("failed to flush future: {e}");
      previous_frame_end = Some(sync::now(device.clone()).boxed());
    }
  }
}


fn create_graphics_pipeline(
  device: &Arc<Device>,
  swapchain: &Arc<Swapchain>,
  vs: &Arc<ShaderModule>,
  fs: &Arc<ShaderModule>
)
-> Arc<GraphicsPipeline>
{
  // Provide defintion of a Vertex Struct
  // TODO: Figure out a way to declare this outside main
  #[derive(BufferContents, Vertex)]
  #[repr(C)]
  struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
  }

  // We want a Rectangle using 4 Vertices. Use `TriangleStrip` instead of `TriangleList`
  let input_assembly_state = InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip);
  //input_assembly_state.topology(PrimitiveTopology::TriangleStrip);

  GraphicsPipeline::start()
    .render_pass(PipelineRenderingCreateInfo {
      color_attachment_formats: vec![Some(swapchain.image_format())],
      ..Default::default()
     })
    .vertex_input_state(Vertex::per_vertex())
    .input_assembly_state(input_assembly_state)
    .vertex_shader(vs.entry_point("main").unwrap(), ())
    .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
    .fragment_shader(fs.entry_point("main").unwrap(), ())
    .build(device.clone())
    .unwrap()
}


fn create_swapchain(
  device: &Arc<Device>,
  surface: &Arc<vulkano::swapchain::Surface>
) 
-> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) 
{
  let surface_capabilities = device
    .physical_device()
    .surface_capabilities(surface, Default::default())
    .unwrap();

  let image_format = Some(
    device
      .physical_device()
      .surface_formats(surface, Default::default())
      .unwrap()[0]
      .0,
  );
  let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

  Swapchain::new(
    device.clone(),
    surface.clone(),
    SwapchainCreateInfo{
      min_image_count: surface_capabilities.min_image_count,
      image_format,
      image_extent: window.inner_size().into(),
      image_usage: ImageUsage::COLOR_ATTACHMENT,
      composite_alpha: surface_capabilities
        .supported_composite_alpha
        .into_iter()
        .next()
        .unwrap(),

      ..Default::default()
    },
  )
  .unwrap()
}


fn create_device(
  instance: &Arc<Instance>, 
  surface: &Arc<vulkano::swapchain::Surface>
) 
-> (Arc<Device>, impl ExactSizeIterator<Item = Arc<vulkano::device::Queue>>) 
{
  // Required extensions for this application
  let mut device_extensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::empty()
  };
  
  // Pick best suited physical device
  let (physical_device, queue_family_index) = pick_physical_device(
    instance, 
    &device_extensions, 
    surface);

  // Log to terminal: information about selected physical device
  println!(
    "\nUsing device: {} (type: {:?})",
    physical_device.properties().device_name,
    physical_device.properties().device_type,
  );
  println!(
    "\nQueue Family Properties: {:?}\n",
    physical_device.queue_family_properties()[queue_family_index as usize],
  );

  // If physical device doesn't have Vulkan Version 1.3 we need `khr_dynamic_rendering` extension
  if physical_device.api_version() < Version::V1_3 {
    device_extensions.khr_dynamic_rendering = true;
  }

  // Create the Vulkan Device object
  Device::new(
    physical_device,
    DeviceCreateInfo {
      enabled_extensions: device_extensions,
      enabled_features: Features {
        dynamic_rendering: true,
        ..Features::empty()
      },
      queue_create_infos: vec![QueueCreateInfo {
        queue_family_index,
        ..Default::default()
      }],

      ..Default::default()
    },
  )
  .unwrap()
}


fn pick_physical_device(
  instance: &Arc<Instance>, 
  device_extensions: &DeviceExtensions,
  surface: &Arc<vulkano::swapchain::Surface>
) 
-> (Arc<vulkano::device::physical::PhysicalDevice>, u32) 
{
  instance.enumerate_physical_devices()
    .unwrap()
    .filter(|d| { // Ensure device has Vulkan version 1.3 or khr_dynamic_rendering extension
      d.api_version() >= Version::V1_3 || d.supported_extensions().khr_dynamic_rendering
    })
    .filter(|d| { // Ensure device has the further extensions we need
      d.supported_extensions().contains(device_extensions)
    })
    .filter_map(|d| { // Need a queue that has graphics capabilities
      d.queue_family_properties()
        .iter()
        .enumerate()
        .position(|(i, q)| { // Select a queue family with graphics operation support.
          q.queue_flags.intersects(QueueFlags::GRAPHICS)
            && d.surface_support(i as u32, &surface).unwrap_or(false)
        })
        .map(|i| (d, i as u32)) // Return the device and index of that queue family to `filter_map`
    })
    .min_by_key(|(d, _)| { // Pick preferred device type of those filtered to this point
      match d.properties().device_type { 
        PhysicalDeviceType::DiscreteGpu => 0,
        PhysicalDeviceType::IntegratedGpu => 1,
        PhysicalDeviceType::VirtualGpu => 2,
        PhysicalDeviceType::Cpu => 3,
        PhysicalDeviceType::Other => 4,
        _ => 5,
      }
    })
    .expect("no suitable physical device found")
}


fn window_size_dependent_setup(
  images: &[Arc<SwapchainImage>],
  viewport: &mut Viewport,
)
-> Vec<Arc<ImageView<SwapchainImage>>> 
{
  let dimensions = images[0].dimensions().width_height();
  viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

  images
    .iter()
    .map(|image| ImageView::new_default(image.clone()).unwrap())
    .collect::<Vec<_>>()
}
