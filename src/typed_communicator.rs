use std::any::TypeId;

use crate::{
    collective::{CommunicatorCollectives, Root, SystemOperation},
    point_to_point::{Destination, Source},
    topology::{Communicator, SimpleCommunicator},
    traits::{Buffer, BufferMut, Equivalence},
};

/// A typed communicator for MPI operations with data type T.
pub struct TypedCommunicator<'a, T>
where
    T: 'static + Equivalence,
{
    communicator: &'a SimpleCommunicator, // Reference to avoid ownership issues
    phantom: std::marker::PhantomData<T>,
}

// #[derive(Eq, PartialEq, Equivalence, Debug, Clone, Default, PartialOrd, Ord)]
// #[mpi(crate = "crate")]
// struct MyTypeId(u64, u64);

#[derive(Eq, PartialEq, Debug, Clone, Default, PartialOrd, Ord)]
struct MyTypeId(u64, u64);

unsafe impl Equivalence for MyTypeId {
    type Base = u64;

    type Out = crate::datatype::UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        let base = u64::equivalent_datatype();
        Self::Out::contiguous(2, &base)
    }

    fn count(&self) -> crate::Count {
        2
    }
}


impl<'a, T> TypedCommunicator<'a, T>
where
    T: 'static + Equivalence,
{
    /// Creates a new `TypedCommunicator` over type `T`.
    /// 
    /// # Examples
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// let _typed_comm = TypedCommunicator::<f32>::new(&world);
    /// ```
    ///
    /// ```compile_fail
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// // Type mismatch example: i32 communicator used inconsistently
    /// let _typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let val: i32 = 42;
    /// _typed_comm.send(&val, 1, 0); //  This will fail to compile
    /// 
    
    pub fn new(communicator: &'a SimpleCommunicator) -> Self {
        // Validate datatype during construction
        let rank = communicator.rank();
        //let size = communicator.size();

        let local_type = TypeId::of::<T>();
        let local_type: MyTypeId = unsafe { std::mem::transmute(local_type) };


        // Fix the unused variable warning
        let _size = communicator.size();

        // Create a boolean flag instead of using max/min directly on MyTypeId
        let is_same_type = true; // Assume true initially
        let mut result = true;

        // Use all_reduce with logical AND operation
        communicator.all_reduce_into(&is_same_type, &mut result, SystemOperation::logical_and());

        // Before the reduction, broadcast the type from rank 0 and compare locally
        let root_type = if rank == 0 {
            local_type.clone()
        } else {
            MyTypeId::default()
        };
        let mut bcast_type = root_type;
        communicator
            .process_at_rank(0)
            .broadcast_into(&mut bcast_type);

        // Each rank compares its type with the broadcast type
        let is_same_type = bcast_type == local_type;
        let mut result = false;

        // Use all_reduce to collect the AND of all comparisons
        communicator.all_reduce_into(&is_same_type, &mut result, SystemOperation::logical_and());

        // Check the result
        if !result {
            panic!(
                "Rank {}: Datatype mismatch detected. Local type: {:?}, Root type: {:?}",
                rank, local_type, bcast_type
            );
        } else if rank == 0 {
            let all_types = vec![local_type.clone(); _size as usize];
            println!(
                "Rank {}: Datatypes validated successfully. Type: {:?}",
                rank, all_types
            );
        }

        TypedCommunicator {
            communicator,
            phantom: std::marker::PhantomData,
        }
    }


    /// Sends data to the specified destination.
    /// # Examples
    ///
    /// Simple values:
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    /// use mpi::point_to_point::Destination;
    ///
    /// use mpi::topology::Communicator;
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// # if world.size() < 2 { return; }
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let data: f32 = 42.0;
    /// typed_comm.send(&data, 0, 0);
    /// ```
    ///
    /// Arrays work as well:
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// use mpi::topology::Communicator;
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// # if world.size() < 2 { return; }
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let data: [f32; 3] = [1.0, 2.0, 3.0];
    /// typed_comm.send(&data, 0, 0);
    /// ```
    ///
    /// Even nested arrays:
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// use mpi::topology::Communicator;
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// # if world.size() < 2 { return; }
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let data: [[f32; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// typed_comm.send(&data, 0, 0);
    /// ```
    ///
    /// # Invalid Uses
    ///
    /// Different types won't compile:
    ///
    /// ```compile_fail
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    /// use mpi::point_to_point::Destination;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let data: i32 = 42;
    /// typed_comm.send(&data, 0, 0); //  Type mismatch: i32 vs f32
    /// ```
    ///
    /// ```compile_fail
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let data: [i32; 3] = [1, 2, 3];
    /// typed_comm.send(&data, 0, 0); //  Type mismatch: i32 vs f32
    /// ```
    /// ```should_panic
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    /// use mpi::topology::Communicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    ///
    /// let value = 3.14_f32;
    /// let invalid_rank = world.size(); // out of bounds
    /// typed_comm.send(&value, invalid_rank, 0); //  should panic
    /// ```
    /// Works for both single values and slices.
    ///
    /// # Arguments
    /// * `data` - The data to send (can be a single value or slice)
    /// * `destination` - Rank of the destination process
    /// * `tag` - Message tag
    /// 
    pub fn send<B>(&self, data: &B, destination: i32, tag: i32)
    where
        B: ?Sized + Buffer<Base = T>,
    {
        self.communicator
            .process_at_rank(destination)
            .send_with_tag(data, tag);
    }

    /// Receives data from the specified source.
    /// Works for both single values and slices.
    ///
    /// # Arguments
    /// * `buffer` - Buffer to receive the data into (can be a single value or slice)
    /// * `source` - Rank of the source process
    /// * `tag` - Message tag
   /// # Examples
    ///
    /// Simple values:
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    /// use mpi::topology::Communicator; // Required for `.size()`
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// # if world.size() < 2 { return; }
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let mut buffer: f32 = 0.0;
    /// typed_comm.receive(&mut buffer, 0, 0);
    /// ```
    ///
    /// Arrays work as well:
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    /// use mpi::topology::Communicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// # if world.size() < 2 { return; }
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let mut buffer: [f32; 3] = [0.0; 3];
    /// typed_comm.receive(&mut buffer, 0, 0);
    /// ```
    ///
    /// Even nested arrays:
    ///
    /// ```
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    /// use mpi::topology::Communicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// # if world.size() < 2 { return; }
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let mut buffer: [[f32; 2]; 3] = [[0.0; 2]; 3];
    /// typed_comm.receive(&mut buffer, 0, 0);
    /// ```
    ///
    /// # Invalid Uses
    ///
    /// Different types won't compile:
    ///
    /// ```compile_fail
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let mut buffer: i32 = 0;
    /// typed_comm.receive(&mut buffer, 0, 0); //  Type mismatch: i32 vs f32
    /// ```
    /// ```compile_fail
    /// use mpi::initialize;
    /// use mpi::typed_communicator::TypedCommunicator;
    ///
    /// let universe = initialize().unwrap();
    /// let world = universe.world();
    /// let typed_comm = TypedCommunicator::<f32>::new(&world);
    /// let mut buffer: [[i32; 2]; 3] = [[0; 2]; 3];
    /// typed_comm.receive(&mut buffer, 0, 0); //  Type mismatch: [[i32; 2]; 3] vs f32
    /// ```
    ///
    pub fn receive<B>(&self, buffer: &mut B, source: i32, tag: i32)
    where
        B: ?Sized + BufferMut<Base = T>,
    {
        self.communicator
            .process_at_rank(source)
            .receive_into_with_tag(buffer, tag);
    }
}
