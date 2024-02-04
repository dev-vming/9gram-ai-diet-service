import styles from './loadingbar.module.css';

export const LoadingBar = () => {
  return (
    <div className={styles.wrapper}>
      <div className={styles.loaderOuter}>
        <div className={styles.loaderInner}>
          <i className='fas fa-utensils' aria-hidden='true' />
          <i
            className='fas fa-robot'
            style={{ color: '#74C0FC' }}
            aria-hidden='true'
          />
          <i className='fas fa-cookie-bite' aria-hidden='true' />
        </div>
      </div>
      <h1 className={styles.title}>
        <span className={styles.titleSpan}>LOADING</span>
      </h1>
    </div>
  );
};
