"""
    이벤트 정의
    WAS : dict : OOM, SQL, EXT, EXC
    DB : dict : BUFF, ENQ, LATCH, REDO, RAC
    OS : dict : CPU, MEM, DISK, NETOL, NETER

OOM = "out of memory"
SQL = "sql delay"
RMF = "remote call delay"
INF = "increase in inflow"
EXC = "exception"

BUFF = "buffer cache and io lock" # latch: cache buffers chains,read by other session
ENQ = "enqueue lock" #enq: TX - row lock contention, enq: TX - index contention, enq: SQ - contention
LATCH = "latch lock" # latch: shared pool, library cache: mutex X, enq: SQ - contention, library cache pin, cursor: pin S wait on X
REDO = "redo lock"# log file switch completion, log file switch (checkpoint incomplete), log file switch (archiving needed)
RAC = "rac condition lock" # excel 참고

CPU = "cpu overload"
MEM = "memory lack"
DISK = "disk lack"
NETOL = "network overload"
NETER = "network error"

"""
OOM = "OOM"
SQL = "SQL"
RMF = "RMF"
INF = "INF"
EXC = "EXC"

BUFF = "BUFF"
ENQ = "ENQ"
LATCH = "LATCH"
REDO = "REDO"
RAC = "RAC"

CPU = "CPU"
MEM = "MEM"
DISK = "DISK"
NETOL = "NETOL"
NETER = "NETER"

EVENT_DOC = {
    "OOM":"Delays in service is expected due to out of memory issues",
    "RMF":"Delays in service is expected due to external call delays",
    "INF":"Delays in service is expected due to increased inflows",
    "EXC":"Delays in service is expected due to an unexpected exception",

    "BUFF":"Delays in query performance is expected due to inefficient sql",
    "ENQ":"Delays in query performance is expected due to enqueue lock contention",
    "LATCH":"Delays in query performance is expected due to latch contention",
    "REDO":"Delays in query performance is expected due to contention in redo",
    "RAC":"Delays in query performance is expected due to contention in the rac environment",

    "CPU":"Overload of cpu is expected",
    "MEM":"Out of memory is expected",
    "DISK":"Insufficient disk space is expected",
    "NETOL":"Network overload is expected",
    "NETER":"Network error is expected",

    "TXND":"Delays is expected"
}

WAS_ERRORS = [OOM, SQL, RMF, INF, EXC]
DB_ERRORS = [BUFF, ENQ, LATCH, REDO, RAC]
OS_ERRORS = [CPU, MEM, DISK, NETOL, NETER]

WAS = {
"active_db_conn_count": [OOM, SQL, INF],
"active_tx_count": [OOM, SQL, RMF, INF],
"call_count": [],
"cpu_time": [],
"cpu_usage": [],
"extcall_count": [RMF],
"extcall_time": [RMF],
"fail_count": [EXC],
"fetch_count": [OOM],
"fetch_time": [OOM],
"file_count": [],
"gc_count": [OOM],
"gc_time": [OOM],
"heap_usage": [OOM],
"prepare_count": [],
"prepare_time": [],
"response_time": [OOM, SQL, INF],
"socket_count": [],
"sql_count": [SQL],
"sql_time": [SQL],
"thread_count": [],
"tps": [INF]
}

DB = {
"active_sessions": [],
"buffer_busy_waits": [BUFF],
"concurrency_wait_time": [],
"consistent_gets": [],
"cpu_used_by_this_session": [],
"cursor_pin_s_wait_on_x": [LATCH],
"db_block_changes": [],
"db_block_gets": [],
"db_file_scattered_read": [BUFF],
"db_file_sequential_read": [BUFF],
"db_time": [],
"enq_sq_contention": [ENQ],
"enq_tx_index_contention": [ENQ],
"enq_tx_row_lock_contention": [ENQ],
"enqueue_requests": [],
"enqueue_waits": [],
"execute_count": [],
"file_io_service_time": [],
"file_io_wait_time": [],
"free_buffer_waits": [],
"gc_buffer_busy_acquire": [RAC],
"gc_buffer_busy_release": [RAC],
"gc_cr_block_busy": [RAC],
"gc_cr_blocks_received": [],
"gc_cr_multi_block_request": [RAC],
"gc_cr_request": [RAC],
"gc_current_block_busy": [RAC],
"gc_current_blocks_received": [],
"gc_current_multi_block_request": [RAC],
"gc_current_request": [RAC],
"global_enqueue_gets_async": [],
"global_enqueue_gets_sync": [],
"latch_cache_buffers_chains": [BUFF],
"latch_shared_pool": [LATCH],
"library_cache_lock": [LATCH],
"library_cache_mutex_x": [LATCH],
"library_cache_pin": [LATCH],
"lock_waiting_sessions": [],
"log_buffer_space": [REDO],
"log_file_sequential_read": [],
"log_file_switch_archiving_needed": [REDO],
"log_file_switch_checkpoint_incomplete": [REDO],
"log_file_switch_completion": [REDO],
"log_file_sync": [REDO],
"non_idle_wait_time": [],
"parse_time_elapsed": [],
"physical_reads": [],
"physical_reads_direct": [],
"physical_writes": [],
"physical_writes_direct": [],
"read_by_other_session": [BUFF],
"recursive_calls": [],
"redo_size": [],
"row_cache_lock": [],
"session_logical_reads": [BUFF],
"user_calls": [],
"user_commits": [],
"user_rollbacks": []
}

OS = {
"cpu_idle": [CPU],
"cpu_system": [CPU],
"cpu_usage": [CPU],
"cpu_usage_max": [CPU],
"cpu_user": [CPU],
"disk_usage": [DISK],
"memory_usage": [MEM],
"memory_used": [MEM],
"network": [NETOL],
"phy_free": [MEM],
"phy_total": [MEM],
"rx_bytes_delta": [NETOL],
"rx_discards_delta": [NETOL],
"rx_errors_delta": [NETER],
"rx_pkts_delta": [NETOL],
"swap_free": [MEM],
"swap_total": [MEM],
"swap_used": [MEM],
"tx_bytes_delta": [NETOL],
"tx_discards_delta": [NETOL],
"tx_errors_delta": [NETER],
"tx_pkts_delta": [NETOL]
}

