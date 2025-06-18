use crate::traits::*;
use crate::topology::{Communicator, Rank};
use crate::point_to_point::Source;
use crate::collective::CommunicatorCollectives;
use std::collections::VecDeque;
use std::time::Duration;

/// A dynamic buffer that can hold different data types
#[derive(Debug, Clone)]
pub enum DynamicBuffer {
    /// 32-bit floating point vector
    F32(Vec<f32>),
    /// 64-bit floating point vector  
    F64(Vec<f64>),
    /// 32-bit integer vector
    I32(Vec<i32>),
    /// 64-bit integer vector
    I64(Vec<i64>),
}

impl DynamicBuffer {
    /// Returns the length of the buffer
    pub fn len(&self) -> usize {
        match self {
            DynamicBuffer::F32(v) => v.len(),
            DynamicBuffer::F64(v) => v.len(),
            DynamicBuffer::I32(v) => v.len(),
            DynamicBuffer::I64(v) => v.len(),
        }
    }

    /// Returns the type identifier for the buffer
    pub fn type_id(&self) -> u8 {
        match self {
            DynamicBuffer::F32(_) => 0,
            DynamicBuffer::F64(_) => 1,
            DynamicBuffer::I32(_) => 2,
            DynamicBuffer::I64(_) => 3,
        }
    }
}

/// Header for messages containing data type and count information
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageHeader {
    /// The data type identifier
    pub data_type: u8,
    /// The number of elements
    pub count: u64,
}

impl MessageHeader {
    /// Converts the header to a byte array
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = vec![self.data_type];
        buffer.extend_from_slice(&self.count.to_le_bytes());
        buffer
    }

    /// Creates a header from a byte array
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 9 {
            return Err(format!("Invalid header length: expected 9, got {}", bytes.len()));
        }
        let data_type = bytes[0];
        let mut count_bytes = [0u8; 8];
        count_bytes.copy_from_slice(&bytes[1..9]);
        let count = u64::from_le_bytes(count_bytes);
        Ok(MessageHeader { data_type, count })
    }
}

/// Result containing data received from other ranks
pub struct TwoSidedResult {
    /// The ranks that sent data
    pub from_ranks: Vec<Rank>,
    /// The data received from each rank
    pub from_data: Vec<DynamicBuffer>,
}

struct SegmentedBuffer<T> {
    data: VecDeque<T>,
    #[allow(dead_code)]
    chunk_size: usize,
}

impl<T> SegmentedBuffer<T> {
    fn new(chunk_size: usize) -> Self {
        Self {
            data: VecDeque::new(),
            chunk_size,
        }
    }

    fn push(&mut self, item: T) {
        self.data.push_back(item);
    }

    fn extract_all(self) -> Vec<T> {
        self.data.into()
    }
}

/// Builds a two-sided communication pattern with immediate barrier synchronization
/// 
/// This function sends data to specified ranks and receives data from any ranks,
/// then synchronizes with a barrier to ensure all communication is complete.
pub fn build_two_sided_ibarrier<C: Communicator>(
    comm: &C,
    to_ranks: &[Rank],
    to_data: &[DynamicBuffer],
) -> Result<TwoSidedResult, Box<dyn std::error::Error>> {
    let tag_header = 100;
    let tag_data = 101;

    assert_eq!(to_ranks.len(), to_data.len(), "Mismatched ranks and data");

    let mut seg_ranks = SegmentedBuffer::new(4);
    let mut seg_data = SegmentedBuffer::new(4);
    
    // Pre-allocate and prepare all buffers to avoid borrowing issues
    let mut header_buffers = Vec::new();
    let mut data_buffers = Vec::new();

    for (_i, _target_rank) in to_ranks.iter().enumerate() {
        let data = &to_data[_i];
        let header = MessageHeader {
            data_type: data.type_id(),
            count: data.len() as u64,
        };
        let header_bytes = header.to_bytes();
        header_buffers.push(header_bytes);

        let raw_bytes: Vec<u8> = match data {
            DynamicBuffer::F32(vec) => bytemuck::cast_slice(vec).to_vec(),
            DynamicBuffer::F64(vec) => bytemuck::cast_slice(vec).to_vec(),
            DynamicBuffer::I32(vec) => bytemuck::cast_slice(vec).to_vec(),
            DynamicBuffer::I64(vec) => bytemuck::cast_slice(vec).to_vec(),
        };
        data_buffers.push(raw_bytes);
    }

    // Use mpi::request::scope to handle the lifetime properly
    crate::request::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let mut all_requests = Vec::new();
        
        // Now send all the data using the pre-allocated buffers
        for (i, &target_rank) in to_ranks.iter().enumerate() {
            // Send header
            all_requests.push(
                comm.process_at_rank(target_rank)
                    .immediate_synchronous_send_with_tag(scope, &header_buffers[i], tag_header),
            );

            // Send data
            all_requests.push(
                comm.process_at_rank(target_rank)
                    .immediate_synchronous_send_with_tag(scope, &data_buffers[i], tag_data),
            );
        }

        let mut barrier_started = false;
        let mut barrier_request = None;
        let start_time = std::time::Instant::now();
        let timeout = Duration::from_secs(30);

        loop {
            if start_time.elapsed() > timeout {
                return Err("Communication timeout - possible deadlock".into());
            }

            if let Some(status) = comm.any_process().immediate_probe_with_tag(tag_header) {
                let (header_data, _) = comm.process_at_rank(status.source_rank())
                    .receive_vec_with_tag(tag_header);
                let header = MessageHeader::from_bytes(&header_data)?;

                let received_data = match header.data_type {
                    0 => {
                        let (data, _) = comm.process_at_rank(status.source_rank())
                            .receive_vec_with_tag(tag_data);
                        DynamicBuffer::F32(data)
                    }
                    1 => {
                        let (data, _) = comm.process_at_rank(status.source_rank())
                            .receive_vec_with_tag(tag_data);
                        DynamicBuffer::F64(data)
                    }
                    2 => {
                        let (data, _) = comm.process_at_rank(status.source_rank())
                            .receive_vec_with_tag(tag_data);
                        DynamicBuffer::I32(data)
                    }
                    3 => {
                        let (data, _) = comm.process_at_rank(status.source_rank())
                            .receive_vec_with_tag(tag_data);
                        DynamicBuffer::I64(data)
                    }
                    _ => return Err("Unknown data type".into()),
                };

                seg_ranks.push(status.source_rank());
                seg_data.push(received_data);
            }

            if !barrier_started {
                // Test all requests by draining and recreating the vector
                let mut remaining_requests = Vec::new();
                
                for req in all_requests.drain(..) {
                    match req.test() {
                        Ok(_) => {
                            // Request completed, don't add it back
                        }
                        Err(req) => {
                            // Request not complete yet, add it back
                            remaining_requests.push(req);
                        }
                    }
                }
                
                all_requests = remaining_requests;
                
                if all_requests.is_empty() {
                    barrier_request = Some(comm.immediate_barrier());
                    barrier_started = true;
                }
            } else if let Some(b_req) = barrier_request.take() {
                match b_req.test() {
                    Ok(_) => break,
                    Err(req) => barrier_request = Some(req),
                }
            }

            std::thread::sleep(Duration::from_millis(1));
        }

        Ok(())
    })?;

    Ok(TwoSidedResult {
        from_ranks: seg_ranks.extract_all(),
        from_data: seg_data.extract_all(),
    })
}