import styles from './LoadingBar.module.css';

export const LoadingBar = () => {
  return (
    <div className={styles.wrapper}>
      <div className={styles.loaderOuter}>
        <div className={styles.loaderInner}>
          <i className='fa fa-ellipsis-h' aria-hidden='true'></i>
        </div>
      </div>
      <h1 className={styles.title}>
        <span className={styles.titleSpan}>LOADING</span>
      </h1>
    </div>
  );
};
