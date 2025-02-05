use crate::topology::SimpleCommunicator;
use crate::datatype::Equivalence;
use crate::raw::AsRaw;
use crate::topology::Communicator;
use crate::collective::CommunicatorCollectives;
use crate::point_to_point::{Destination, Source};
/// A typed communicator for MPI operations with data type T.
pub struct TypedCommunicator<'a, T>
where
    T: Equivalence,
{
    communicator: &'a SimpleCommunicator, // Reference to avoid ownership issues
    phantom: std::marker::PhantomData<T>,
}
impl<'a, T> TypedCommunicator<'a, T>
where
    T: Equivalence,
{
    /// Creates a new `TypedCommunicator` over type `T`.
    pub fn new(communicator: &'a SimpleCommunicator) -> Self {
        // Validate datatype during construction
        let rank = communicator.rank();
        let size = communicator.size();
        let local_datatype = T::equivalent_datatype().as_raw();
        // Collect datatype info across ranks
        let mut all_datatypes = vec![local_datatype; size as usize];
        communicator.all_gather_into(&local_datatype, &mut all_datatypes);
        // Check congruence
        let is_congruent = all_datatypes.iter().all(|&dt| dt == local_datatype);
        if !is_congruent {
            panic!(
                "Rank {}: Datatype mismatch detected among ranks: {:?}",
                rank, all_datatypes
            );
        } else if rank == 0 {
            println!(
                "Rank {}: Datatypes validated successfully: {:?}",
                rank, all_datatypes
            );
        }
        TypedCommunicator {
            communicator,
            phantom: std::marker::PhantomData,
        }
    }
   /// Sends a single value to the specified destination.
   pub fn send_value(&self, data: &T, destination: i32, _tag: i32) {
        // Type-checking for `send_value`
        if T::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `send_value`: Cannot send data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<T>(),
                std::any::type_name::<T>()
            );
        }
        self.communicator
            .process_at_rank(destination)
            .send(data);
    }
    /// Sends a slice of values to the specified destination.
    pub fn send_slice<U>(&self, data: &[U], destination: i32, _tag: i32)
    where
        U: Equivalence,
    {
        // Type-checking for `send_slice`
        if U::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `send_slice`: Cannot send data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<U>(),
                std::any::type_name::<T>()
            );
        }
        self.communicator
            .process_at_rank(destination)
            .send(data);
    }
    /// Receives a single value from the specified source.
    pub fn receive_value(&self, buffer: &mut T, source: i32, _tag: i32) {
        // Type-checking for `receive_value`
        if T::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `receive_value`: Cannot receive data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<T>(),
                std::any::type_name::<T>()
            );
        }
        self.communicator
            .process_at_rank(source)
            .receive_into(buffer);
    }
    /// Receives a slice of values from the specified source.
    pub fn receive_slice<U>(&self, buffer: &mut [U], source: i32, _tag: i32)
    where
        U: Equivalence,
    {
        // Type-checking for `receive_slice`
        if U::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `receive_slice`: Cannot receive data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<U>(),
                std::any::type_name::<T>()
            );
        }
        self.communicator
            .process_at_rank(source)
            .receive_into(buffer);
    }
}