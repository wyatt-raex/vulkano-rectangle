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
      input_assembly::InputAssemblyState,
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
  Version, VulkanLibrary,
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

  let event_loop = EventLoop::new();
  let surface = WindowBuilder::new()
    .build_vk_surface(&event_loop, instance.clone())
    .unwrap();

  let (device, mut queues) = create_device(&instance, &surface);
  let queue = queues.next().unwrap();

  let (mut swapchain, images) = create_swapchain(&device, &surface);
  let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
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
