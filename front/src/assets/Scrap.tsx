import styles from '@components/pages/ai-analyze/box.module.css';

interface Props {
  scrap: boolean;
}

export const Scrap = ({ scrap }: Props) => {
  return (
    <svg
      xmlns='http://www.w3.org/2000/svg'
      viewBox='0 0 24 24'
      fill='none'
      strokeWidth='2'
      stroke='currentColor'
      className={scrap ? `w-6 h-6 ${styles.scrapped}` : 'w-6 h-6'}
      width='13'
    >
      <path
        strokeLinecap='round'
        strokeLinejoin='round'
        d='M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z'
      />
    </svg>
  );
};
