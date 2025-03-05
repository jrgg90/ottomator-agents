-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    user_id uuid references auth.users(id) on delete cascade,
    url varchar not null,
    marketplace varchar not null check (marketplace in ('amazon', 'ebay', 'etsy')),
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Create an index on marketplace
create index idx_site_pages_marketplace on site_pages (marketplace); 

-- Indexes for user_conversations
create index idx_user_conversations_user_id on user_conversations(user_id);
create index idx_user_conversations_session_id on user_conversations(session_id);
create index idx_user_conversations_marketplace on user_conversations(marketplace);

-- Index for conversation_images
create index idx_conversation_images_conversation_id on conversation_images(conversation_id);


-- Create a function to search for documentation chunks
create or replace function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  marketplace_filter varchar default null,  -- Nuevo parámetro para filtrar por marketplace
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  marketplace varchar,  -- Añadido marketplace a los resultados
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    marketplace,  -- Incluir marketplace en los resultados
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
    AND (marketplace_filter IS NULL OR marketplace = marketplace_filter)  -- Filtro condicional por marketplace
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

create table user_conversations (
    id bigserial primary key,
    user_id uuid references auth.users(id) on delete cascade,
    session_id text not null,  -- Para agrupar conversaciones relacionadas
    question text not null,
    answer text not null,
    marketplace varchar,  -- Para filtrar conversaciones por marketplace
    sources jsonb,  -- Para guardar las fuentes usadas en la respuesta
    created_at timestamp with time zone default timezone('utc', now()) not null
);

-- Tabla para almacenar imágenes de conversaciones
create table conversation_images (
    id bigserial primary key,
    conversation_id bigint references user_conversations(id) on delete cascade,
    image_url text not null,  -- URL de la imagen en almacenamiento (ej. Supabase Storage)
    image_analysis text,      -- Análisis o descripción de la imagen (generado por IA)
    created_at timestamp with time zone default timezone('utc', now()) not null
);

-- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- Enable RLS on the tables
alter table site_pages enable row level security;
alter table user_conversations enable row level security;
alter table conversation_images enable row level security;


-- Create a policy that allows authenticated users to read
create policy "Allow authenticated users to read"
  on site_pages
  for select
  using (auth.role() = 'authenticated');

create policy "Users can only access their own conversations"
    on user_conversations
    for select
    using (auth.uid() = user_id);

-- Política para que los usuarios solo puedan ver sus propias imágenes
create policy "Users can only access their own conversation images"
  on conversation_images
  for select
  using (
    conversation_id IN (
      SELECT id FROM user_conversations WHERE user_id = auth.uid()
    )
  );


