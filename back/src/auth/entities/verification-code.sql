CREATE TABLE verification_code (
    verification_code_id uuid PRIMARY KEY,
    email varchar(50),
    code varchar(8),
    created_date timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);