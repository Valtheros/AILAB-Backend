create sequence if not exists storage_project_id_seq as bigint
  minvalue 1000000 maxvalue 4294967294 start with 1000000 no cycle;

create table if not exists storage_accounts (
  owner_user_id text primary key references "user"(id) on delete restrict,
  project_id bigint not null unique default nextval('storage_project_id_seq')
    check (project_id between 1 and 4294967294),
  limit_bytes bigint not null default 21474836480 check (limit_bytes > 0),
  state text not null default 'pending' check (state in ('pending', 'ready', 'error')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists storage_reservations (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null references storage_accounts(owner_user_id) on delete restrict,
  operation_key text not null check (length(operation_key) between 1 and 128),
  kind text not null check (kind in ('upload', 'import', 'label_upload', 'label_publish',
    'train', 'evaluate', 'predict', 'batch_predict', 'index', 'transfer')),
  reserved_bytes bigint not null check (reserved_bytes > 0),
  created_at timestamptz not null default now(),
  released_at timestamptz,
  unique (owner_user_id, operation_key)
);

create index if not exists storage_reservations_active_owner
  on storage_reservations(owner_user_id) where released_at is null;
