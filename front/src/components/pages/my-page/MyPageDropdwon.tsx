import toggle from './mypageedit.module.css';
import { MyPageDropdownProps } from './MypageTypes';
import { DownArrow } from '@assets/DownArrow';

const MyPageDropdown = ({
  items,
  selectedItem,
  onSelectItem,
  toggleDropdown,
  isDropdownVisible,
  style,
}: MyPageDropdownProps) => {
  return (
    <div style={style}>
      <div className={toggle.toggleTitle} onClick={toggleDropdown}>
        <div>{selectedItem}</div>
        <DownArrow className={toggle.arrow} />
      </div>
      {isDropdownVisible && (
        <div className={toggle.dropdown}>
          {items.map((item: string) => (
            <div
              key={item}
              onClick={() => onSelectItem(item)}
              className={selectedItem === item ? toggle.active : ''}
            >
              {item}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MyPageDropdown;
